import os
import cv2
import numpy as np
from ultralytics import YOLO

class VehicleHelmetDetector:
    def __init__(self, config):
        """
        Initializes the YOLOv8 detector.
        
        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.model_path = config.get("models", {}).get("detector_weights", "models/best_detector.pt")
        self.thresholds = config.get("models", {}).get("confidence", {
            "rider": 0.35, "helmet": 0.40, "no_helmet": 0.35, "license_plate": 0.40
        })
        
        self.model = None
        self.has_custom_classes = False
        self._load_model()

    def _load_model(self):
        """Loads the trained YOLOv8 model or standard YOLOv8 weights."""
        if os.path.exists(self.model_path):
            print(f"[Detector] Loading custom weights from {self.model_path}...")
            try:
                self.model = YOLO(self.model_path)
                self.names = self.model.names
                self.has_custom_classes = any(
                    any(k in str(v).lower() for k in ["rider", "helmet", "plate", "license"])
                    for v in self.names.values()
                )
                print(f"[Detector] Custom model loaded successfully with classes: {self.names}")
            except Exception as e:
                print(f"[Detector] Error loading custom model: {e}. Loading standard YOLO weights.")
                self._setup_yolo_model()
        else:
            print(f"[Detector] Custom weights not found at '{self.model_path}'. Loading generic YOLOv8n (COCO).")
            print(f"[Detector] WARNING: COCO mode cannot detect helmets or license plates. "
                  f"Only person+motorcycle rider association is available.")
            self._setup_yolo_model()

    def _setup_yolo_model(self):
        """Initializes standard YOLOv8 model for vehicle and rider detection."""
        try:
            self.model = YOLO("yolov8n.pt")
            self.names = self.model.names
            self.has_custom_classes = False
            print("[Detector] YOLOv8n (COCO) model initialized. Helmet/plate detection unavailable.")
        except Exception as e:
            print(f"[Detector] CRITICAL: Error loading YOLO model: {e}")
            self.model = None
            self.names = {}

    def detect(self, img):
        """
        Runs object detection on the input image.
        
        Args:
            img (numpy.ndarray): Input BGR image.
            
        Returns:
            dict: Categorized detections by class name.
        """
        if img is None or img.size == 0:
            return {"riders": [], "helmets": [], "no_helmets": [], "plates": []}

        if self.model is None:
            print("[Detector] No model loaded — returning empty detections.")
            return {"riders": [], "helmets": [], "no_helmets": [], "plates": []}

        # Run inference
        results = self.model(img, verbose=False)[0]
        boxes = results.boxes
        
        detections = {
            "riders": [],
            "helmets": [],
            "no_helmets": [],
            "plates": []
        }
        
        if self.has_custom_classes:
            # Custom-trained model with rider/helmet/no-helmet/plate classes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                class_name = self.names.get(cls_id, "").lower()
                
                # Assign to categories
                if "rider" in class_name and conf >= self.thresholds.get("rider", 0.35):
                    detections["riders"].append({"box": xyxy, "conf": conf})
                elif "no-helmet" in class_name and conf >= self.thresholds.get("no_helmet", 0.35):
                    detections["no_helmets"].append({"box": xyxy, "conf": conf})
                elif "helmet" in class_name and conf >= self.thresholds.get("helmet", 0.40):
                    detections["helmets"].append({"box": xyxy, "conf": conf})
                elif ("plate" in class_name or "license" in class_name) and conf >= self.thresholds.get("license_plate", 0.40):
                    detections["plates"].append({"box": xyxy, "conf": conf})
        else:
            # Standard YOLO COCO class mapping (0: person, 3: motorcycle)
            # NOTE: COCO mode can only detect persons and motorcycles.
            # It CANNOT detect helmets, no-helmets, or license plates.
            # Rider association (person on motorcycle) is the only inference available.
            persons = []
            motorcycles = []
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                if cls_id == 0 and conf >= 0.35: # person
                    persons.append({"box": xyxy, "conf": conf})
                elif cls_id == 3 and conf >= 0.35: # motorcycle
                    motorcycles.append({"box": xyxy, "conf": conf})
            
            # Associate persons on/near motorcycles as 'riders'
            for motor in motorcycles:
                m_box = motor["box"]
                m_x_center = (m_box[0] + m_box[2]) / 2
                
                for person in persons:
                    p_box = person["box"]
                    p_x_center = (p_box[0] + p_box[2]) / 2
                    
                    if abs(p_x_center - m_x_center) < (m_box[2] - m_box[0]) * 0.7:
                        if p_box[3] > m_box[1] - (m_box[3] - m_box[1]) * 0.3:
                            detections["riders"].append({"box": p_box, "conf": person["conf"]})

            # No helmet detection, no plate detection — these require custom weights.
            # detections["helmets"], detections["no_helmets"], detections["plates"]
            # remain empty, which is the honest result.

        return detections

    def associate_violations(self, detections):
        """
        Associates "no-helmet" detections with the closest "license plate".
        
        Algorithm:
        - For every no-helmet crop, locate the closest plate below it in 2D space.
        - Distance is calculated from the bottom center of the no-helmet box to the top center of the plate box.
        - The plate must be below the no-helmet box (or close to it vertically).
        
        Returns:
            list: A list of dicts, each pairing a rider crop, head/no-helmet crop, and plate crop.
        """
        pairs = []
        used_plates = set()
        
        for no_helmet in detections["no_helmets"]:
            nh_box = no_helmet["box"]
            nh_bottom_center = ((nh_box[0] + nh_box[2]) / 2, nh_box[3])
            
            best_plate = None
            min_dist = float('inf')
            
            for idx, plate in enumerate(detections["plates"]):
                if idx in used_plates:
                    continue
                p_box = plate["box"]
                p_top_center = ((p_box[0] + p_box[2]) / 2, p_box[1])
                
                # Check if plate is below the head/no-helmet detection
                # License plate must have a Y center larger than the head's Y bottom
                if p_top_center[1] > nh_bottom_center[1]:
                    # Compute Euclidean distance
                    dist = np.sqrt((nh_bottom_center[0] - p_top_center[0])**2 + (nh_bottom_center[1] - p_top_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_plate = (idx, plate)
            
            # If a plate is found and it is within reasonable distance
            # (e.g. less than 1.5 times the height of the image to handle varying scales)
            if best_plate is not None:
                idx, plate_info = best_plate
                used_plates.add(idx)
                
                # Find the rider box that contains or is closest to this no-helmet head
                best_rider = None
                r_min_dist = float('inf')
                for rider in detections["riders"]:
                    r_box = rider["box"]
                    # Check overlap or proximity
                    # Check if head is inside rider box
                    if r_box[0] <= nh_box[0] and r_box[1] <= nh_box[1] and r_box[2] >= nh_box[2] and r_box[3] >= nh_box[3]:
                        best_rider = rider
                        break
                    else:
                        r_center = ((r_box[0] + r_box[2])/2, (r_box[1] + r_box[3])/2)
                        nh_center = ((nh_box[0] + nh_box[2])/2, (nh_box[1] + nh_box[3])/2)
                        dist = np.sqrt((r_center[0] - nh_center[0])**2 + (r_center[1] - nh_center[1])**2)
                        if dist < r_min_dist:
                            r_min_dist = dist
                            best_rider = rider
                            
                pairs.append({
                    "no_helmet": no_helmet,
                    "plate": plate_info,
                    "rider": best_rider if best_rider else no_helmet
                })
                
        return pairs

    def draw_annotations(self, img, detections, violations):
        """
        Draws bounding boxes and labels on the image for visualization.
        Red for violations (no-helmet, associated plates), Green for helmet compliance.
        """
        annotated = img.copy()
        
        # Track associated plate boxes to color them red
        violating_plates = [v["plate"]["box"] for v in violations]
        
        # Draw riders
        for r in detections["riders"]:
            box = r["box"]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            cv2.putText(annotated, f"Rider {r['conf']:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw helmets
        for h in detections["helmets"]:
            box = h["box"]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(annotated, "Helmet", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw no-helmets
        for nh in detections["no_helmets"]:
            box = nh["box"]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(annotated, "NO HELMET", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw plates
        for p in detections["plates"]:
            box = p["box"]
            is_violating = any(np.array_equal(box, v_box) for v_box in violating_plates)
            color = (0, 0, 255) if is_violating else (255, 0, 0)
            label = "Plate (VIOLATION)" if is_violating else "Plate"
            
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(annotated, f"{label} {p['conf']:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated
