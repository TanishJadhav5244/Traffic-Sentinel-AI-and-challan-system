import os
import yaml
from ultralytics import YOLO

def create_dataset_yaml(dataset_dir="dataset"):
    """
    Creates the dataset.yaml file required by YOLOv8 for training.
    
    Args:
        dataset_dir (str): Root path to the dataset folder containing train/val splits.
    """
    data_yaml = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test", # optional
        
        # 4-class single model setup
        "names": {
            0: "rider",
            1: "helmet",
            2: "no-helmet",
            3: "license-plate"
        }
    }
    
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    os.makedirs(dataset_dir, exist_ok=True)
    
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"[Trainer] Created dataset configuration at: {yaml_path}")
    return yaml_path

def train_yolo(data_yaml_path, epochs=50, batch_size=16, img_size=640, device="cpu"):
    """
    Loads YOLOv8 nano pre-trained weights and trains on the custom dataset.
    
    Args:
        data_yaml_path (str): Path to dataset.yaml.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        img_size (int): Image size (typically 640 for YOLOv8).
        device (str): "cpu" or "cuda" (for GPU) or "0" (first GPU).
    """
    print(f"[Trainer] Initializing YOLOv8 training on {device}...")
    
    # Initialize YOLOv8 nano model (fast and perfect for edge devices / CPU)
    # Use yolov8s.pt (small) or yolov8m.pt (medium) for higher accuracy if GPU available
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=4,
        save=True,
        project="models/runs",
        name="helmet_plate_detector"
    )
    
    print("[Trainer] Training completed successfully!")
    print(f"[Trainer] Best model weights saved to: models/runs/helmet_plate_detector/weights/best.pt")
    
    # Copy best weights to config's default path
    best_weights = "models/runs/helmet_plate_detector/weights/best.pt"
    target_weights = "models/best_detector.pt"
    
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"[Trainer] Copied best weights to target: {target_weights}")
        
    return results

def evaluate_yolo():
    """
    Validates/Evaluates the trained model on validation set.
    """
    target_weights = "models/best_detector.pt"
    if not os.path.exists(target_weights):
        print(f"[Trainer] Cannot evaluate: Custom weights not found at {target_weights}")
        return
        
    print("[Trainer] Loading custom weights for validation...")
    model = YOLO(target_weights)
    
    # Run validation
    metrics = model.val()
    
    print("\n" + "="*40)
    print("YOLOv8 MODEL METRICS SUMMARY")
    print("="*40)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("="*40)

if __name__ == "__main__":
    # --- Instructions to source dataset ---
    # 1. Download a helmet & plate detection dataset (e.g. from Roboflow or Kaggle).
    #    Example Kaggle dataset: "Helmet Detection Dataset" or "Indian License Plate Dataset"
    #    Combine their annotations and format them as YOLO v8 format.
    # 2. Place files in 'dataset' directory:
    #    dataset/
    #      images/
    #        train/
    #        val/
    #      labels/
    #        train/
    #        val/
    # 3. Set custom training execution:
    
    print("="*60)
    print("YOLOv8 Helmet & Plate Detection Training Script")
    print("="*60)
    print("Note: To run this training, make sure your dataset is formatted in YOLOv8")
    print("text format and placed under a 'dataset/' folder in the workspace.")
    
    import sys
    import torch
    
    # Determine execution device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup dataset folders if executing as a test
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        yaml_path = create_dataset_yaml("dataset")
        train_yolo(yaml_path, epochs=10, batch_size=8, device=device)
        evaluate_yolo()
    else:
        print("\nTo launch training, run this script with '--train' flag:")
        print("  python models/train_detector.py --train")
        print("\nGenerating configuration template dataset.yaml...")
        create_dataset_yaml("dataset")
