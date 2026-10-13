# from roboflow import Roboflow
# rf = Roboflow(api_key="EaGowTbKzBdGHcTc5aVr")
# project = rf.workspace("aina-lorinel-rakotoson").project("sacsia-parking-gjyke")
# version = project.version(1)
# dataset = version.download("yolov11")

import os
from ultralytics import YOLO

def train_model():
    # 1. Charger le modèle pré-entraîné YOLOv11 (Nano est rapide et efficace)
    model = YOLO("modeles/yolo11n.pt")
    dataset_yaml = os.path.join(os.getcwd(), "sacsia-parking-1", "data.yaml")

    print(f"Lancement de l'entraînement avec le fichier : {dataset_yaml}")

    results = model.train(
        data=dataset_yaml,
        epochs=60,                   
        imgsz=640,                  
        batch=16,                   
        device=0,                   
        name="model_for_parking", 
        workers=2
    )

    print("\n--------------------------------------------------")
    print("Entraînement terminé avec succès !")
    print("Ton modèle entraîné est sauvegardé à l'emplacement :")
    print("runs/detect/model_for_parking/weights/best.pt")
    print("--------------------------------------------------")

if __name__ == '__main__':
    train_model()