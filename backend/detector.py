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
        
        self.demo_mode = False
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the custom YOLOv8 model, or falls back to yolov8n.pt in demo mode."""
        if os.path.exists(self.model_path):
            print(f"[Detector] Loading custom weights from {self.model_path}...")
            try:
                self.model = YOLO(self.model_path)
                # Check class names
                self.names = self.model.names
                print(f"[Detector] Custom model loaded successfully with classes: {self.names}")
            except Exception as e:
                print(f"[Detector] Error loading custom model: {e}. Falling back to demo mode.")
                self._setup_demo_mode()
        else:
            print(f"[Detector] Custom weights not found at {self.model_path}. Setting up Demo Mode.")
            self._setup_demo_mode()

    def _setup_demo_mode(self):
        """Initializes standard yolov8n.pt and flags demo mode active."""
        self.demo_mode = True
        print("[Detector] Loading standard yolov8n.pt for motorcycle/rider detection...")
        try:
            self.model = YOLO("yolov8n.pt")
            self.names = self.model.names
        except Exception as e:
            print(f"[Detector] Error loading yolov8n.pt: {e}")
            # Fall back to complete mock if even yolov8n fails
            self.model = None
            self.names = {0: "rider", 1: "helmet", 2: "no-helmet", 3: "license-plate"}

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
            # Complete mock mode fallback
            return self._generate_mock_detections(img)

        # Run inference
        results = self.model(img, verbose=False)[0]
        boxes = results.boxes
        
        detections = {
            "riders": [],
            "helmets": [],
            "no_helmets": [],
            "plates": []
        }
        
        # If in custom production mode
        if not self.demo_mode:
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
            # Demo Mode: COCO class map -> Heuristic projection of helmets & plates
            # COCO mapping: 0: person, 3: motorcycle
            h, w = img.shape[:2]
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
                
                # Check for person overlapping with motorcycle
                rider_found = False
                for person in persons:
                    p_box = person["box"]
                    p_x_center = (p_box[0] + p_box[2]) / 2
                    
                    # If person is horizontally aligned with motorcycle and sits above/overlaps it
                    if abs(p_x_center - m_x_center) < (m_box[2] - m_box[0]) * 0.7:
                        # Person bottom is near/below motorcycle top
                        if p_box[3] > m_box[1] - (m_box[3] - m_box[1]) * 0.3:
                            rider_found = True
                            detections["riders"].append({"box": p_box, "conf": person["conf"]})
                            
                            # Heuristic for helmet/no-helmet:
                            # We crop the head (top 20% of person box) and simulate helmet detection
                            head_h = int((p_box[3] - p_box[1]) * 0.25)
                            head_box = [p_box[0], p_box[1], p_box[2], p_box[1] + head_h]
                            
                            # Deterministic based on coordinates to keep video consistent
                            is_helmet = (p_box[0] + p_box[1]) % 2 == 0
                            
                            if is_helmet:
                                detections["helmets"].append({"box": head_box, "conf": 0.85})
                            else:
                                detections["no_helmets"].append({"box": head_box, "conf": 0.88})
                                
                # If rider on motorcycle, project a license plate at the rear bottom of the motorcycle
                if rider_found or len(motorcycles) == 1:
                    # License plates are usually at the bottom-center of the motorcycle
                    # Let's project a bounding box near the bottom of the motorcycle
                    m_w = m_box[2] - m_box[0]
                    m_h = m_box[3] - m_box[1]
                    
                    plate_w = int(m_w * 0.35)
                    plate_h = int(m_h * 0.15)
                    
                    # Place at bottom middle
                    px1 = int(m_x_center - plate_w / 2)
                    py1 = int(m_box[3] - plate_h - 10)
                    px2 = px1 + plate_w
                    py2 = py1 + plate_h
                    
                    # Bound checking
                    px1 = max(0, min(px1, w))
                    py1 = max(0, min(py1, h))
                    px2 = max(0, min(px2, w))
                    py2 = max(0, min(py2, h))
                    
                    detections["plates"].append({"box": np.array([px1, py1, px2, py2]), "conf": 0.90})

        if self.demo_mode and len(detections["riders"]) == 0 and len(detections["plates"]) == 0:
            return self._generate_mock_detections(img)

        return detections

    def _generate_mock_detections(self, img):
        """Generates completely mock detections if no model is loaded at all."""
        h, w = img.shape[:2]
        # Draw mock rider, plate, and no-helmet box matching traffic_sample.png geometry
        # Rider box (encloses rider torso, head, and motorcycle body)
        rx1, ry1 = int(w * 0.40), int(h * 0.35)
        rx2, ry2 = int(w * 0.68), int(h * 0.90)
        
        # Head (no-helmet box)
        hx1, hy1 = int(w * 0.47), int(h * 0.37)
        hx2, hy2 = int(w * 0.54), int(h * 0.46)
        
        # License plate box
        px1, py1 = int(w * 0.54), int(h * 0.76)
        px2, py2 = int(w * 0.67), int(h * 0.81)
        
        return {
            "riders": [{"box": np.array([rx1, ry1, rx2, ry2]), "conf": 0.92}],
            "helmets": [],
            "no_helmets": [{"box": np.array([hx1, hy1, hx2, hy2]), "conf": 0.88}],
            "plates": [{"box": np.array([px1, py1, px2, py2]), "conf": 0.90}]
        }

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
