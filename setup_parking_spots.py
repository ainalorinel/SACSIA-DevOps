import cv2
import json
import os
import numpy as np

# Configuration
JSON_FILE = "parking_spots.json"
CAMERA_INDEX = 1  # Index de la caméra parking (1 pour USB/filaire externe, 0 pour intégrée)

# Variables globales
current_polygon = []
parking_spots = []

# Charger les zones existantes si le fichier JSON existe déjà
if os.path.exists(JSON_FILE):
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
            parking_spots = data.get("spots", [])
            print(f"✅ {len(parking_spots)} places chargées depuis {JSON_FILE}")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du fichier JSON: {e}")

def mouse_callback(event, x, y, flags, param):
    global current_polygon, parking_spots

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        
        # Quand 4 points sont placés, la zone est finalisée
        if len(current_polygon) == 4:
            spot_id = len(parking_spots) + 1
            parking_spots.append({
                "id": spot_id,
                "points": current_polygon.copy()
            })
            print(f"➕ Place #{spot_id} ajoutée : {current_polygon}")
            current_polygon = []

def draw_overlay(frame):
    # Dessiner les places déjà confirmées
    for spot in parking_spots:
        pts = np.array(spot["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Zone semi-transparente bleue
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 128, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Contour vert
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Afficher l'ID au centre de la place
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, f"P{spot['id']}", (cX - 15, cY + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Dessiner les points en cours de placement
    if len(current_polygon) > 0:
        for pt in current_polygon:
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
        if len(current_polygon) > 1:
            pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra à l'index {CAMERA_INDEX}")
        return

    cv2.namedWindow("SACSIA - Tracage des places de Parking")
    cv2.setMouseCallback("SACSIA - Tracage des places de Parking", mouse_callback)

    print("\n--- INSTRUCTIONS DE TRACAGE ---")
    print("• Clic Gauche : Placer les 4 sommets d'une place")
    print("• 's' : Sauvegarder dans 'parking_spots.json' et Quitter")
    print("• 'z' : Annuler la dernière action")
    print("• 'r' : Recommencer à zéro (Effacer toutes les zones)")
    print("• 'q' / Esc : Quitter sans sauvegarder\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur de flux vidéo...")
            break

        draw_overlay(frame)

        # Instructions sur la vidéo
        cv2.putText(frame, f"Places tracees : {len(parking_spots)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Appuyez sur 's' pour Sauvegarder", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("SACSIA - Tracage des places de Parking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Sauvegarde dans le fichier JSON
            with open(JSON_FILE, "w") as f:
                json.dump({"spots": parking_spots}, f, indent=4)
            print(f"\n✅ {len(parking_spots)} places sauvegardées avec succès dans '{JSON_FILE}' !")
            break
        elif key == ord('z'):
            if current_polygon:
                current_polygon.pop()
            elif parking_spots:
                parking_spots.pop()
                print("↩️ Dernière place supprimée")
        elif key == ord('r'):
            parking_spots.clear()
            current_polygon.clear()
            print("🔄 Réinitialisation de toutes les places")
        elif key == ord('q') or key == 27:
            print("❌ Annulation (aucune modification enregistrée)")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()