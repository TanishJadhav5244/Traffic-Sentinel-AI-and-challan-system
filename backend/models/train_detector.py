import os
from ultralytics import YOLO

def train_custom_detector(
    data_yaml="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    weights_output_dir="models"
):
    """
    Fine-tunes YOLOv8 model for two-wheeler helmet compliance and license plate detection.
    
    Expected classes in data.yaml:
      0: rider
      1: helmet
      2: no-helmet
      3: license-plate
    """
    if not os.path.exists(data_yaml):
        print(f"[Trainer Error] Dataset configuration file '{data_yaml}' not found.")
        print("Please provide a valid Roboflow YOLOv8 data.yaml file.")
        return False

    os.makedirs(weights_output_dir, exist_ok=True)
    
    print("[Trainer] Initializing YOLOv8n base model for fine-tuning...")
    model = YOLO("yolov8n.pt")

    print(f"[Trainer] Starting training for {epochs} epochs on '{data_yaml}'...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=weights_output_dir,
        name="helmet_plate_yolov8",
        exist_ok=True
    )
    
    best_weights = os.path.join(weights_output_dir, "helmet_plate_yolov8", "weights", "best.pt")
    target_weights = os.path.join(weights_output_dir, "best_detector.pt")
    
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"[Trainer] Success! Trained weights exported to '{target_weights}'.")
        return True
    return False

if __name__ == "__main__":
    train_custom_detector()
