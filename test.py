import cv2
import json
import os
import numpy as np

# Fichier de sortie JSON
JSON_FILE = "parking_spots.json"

# Variables globales
current_polygon = []
parking_spots = []

# Charger les places existantes de manière sécurisée
if os.path.exists(JSON_FILE):
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Gestion flexible : liste directe ou dictionnaire avec clé "spots"
            if isinstance(data, list):
                parking_spots = [s for s in data if isinstance(s, dict)]
            elif isinstance(data, dict):
                spots_list = data.get("spots", [])
                parking_spots = [s for s in spots_list if isinstance(s, dict)]
            print(f"  {len(parking_spots)} places valides chargées depuis {JSON_FILE}")
    except Exception as e:
        print(f" Erreur lors du chargement de {JSON_FILE}: {e}")
        parking_spots = []

def save_to_json():
    """Enregistre la liste des places dans le fichier JSON."""
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(parking_spots, f, indent=4)
    print(f" Configuration sauvegardée dans {JSON_FILE} ({len(parking_spots)} places).")

def mouse_callback(event, x, y, flags, param):
    global current_polygon, parking_spots

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        print(f"Point ajouté : ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            spot_id = len(parking_spots) + 1
            spot_data = {
                "id": spot_id,
                "name": f"Place #{spot_id}",
                "coordinates": current_polygon.copy()
            }
            parking_spots.append(spot_data)
            print(f"  Place #{spot_id} enregistrée avec {len(current_polygon)} sommets.")
            current_polygon = []
            save_to_json()
        else:
            print(" Un polygone doit contenir au moins 3 points !")

def main():
    global current_polygon, parking_spots
    cam_link = "https://192.168.4.3:8080"
    cap = cv2.VideoCapture(cam_link, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "SACSIA - Tracage des places de parking"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(" OUTIL DE SÉLECTION DES PLACES DE PARKING")
    print(" Clic GAUCHE : Placer un point du polygone")
    print(" Clic DROIT  : Valider et fermer la place courante")
    print(" Touche 'z'  : Annuler le dernier point du polygone")
    print(" Touche 'u'  : Supprimer la dernière place enregistrée")
    print(" Touche 'c'  : Effacer TOUTES les places")
    print(" Touche 'q'  : Quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur d'accès à la caméra.")
            break

        overlay = frame.copy()

        # 1. Dessiner les places déjà validées en toute sécurité
        for spot in parking_spots:
            # Récupération sécurisée des coordonnées (gère aussi l'ancienne clé 'points')
            coords = spot.get("coordinates") or spot.get("points")
            if not coords:
                continue

            pts = np.array(coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, f"#{spot.get('id', 1)}", (cx - 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Dessiner le polygone en cours de création
        if len(current_polygon) > 0:
            pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            for pt in current_polygon:
                cv2.circle(overlay, tuple(pt), 4, (0, 165, 255), -1)

        cv2.imshow(window_name, overlay)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            if current_polygon:
                current_polygon.pop()
                print("↩️ Dernier point annulé.")
        elif key == ord('u'):
            if parking_spots:
                removed = parking_spots.pop()
                save_to_json()
                print(f"🗑️ Place #{removed.get('id', '?')} supprimée.")
        elif key == ord('c'):
            parking_spots = []
            current_polygon = []
            save_to_json()
            print("🧹 Toutes les places ont été réinitialisées.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()