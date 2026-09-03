import os
import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Paths to the two specialist models (downloaded by download_weights.py)
# ──────────────────────────────────────────────────────────────────────────────
_PLATE_MODEL_PATH   = os.path.join("models", "plate_detector.pt")
_HELMET_MODEL_PATH  = os.path.join("models", "helmet_detector.pt")


class VehicleHelmetDetector:
    def __init__(self, config):
        """
        Initialises the YOLOv8-based detector.

        Loading strategy (in order of preference):
          1. Combined single model  -> models/best_detector.pt
          2. Dual specialist models -> models/plate_detector.pt + models/helmet_detector.pt
          3. COCO yolov8n.pt        -> persons/motorcycles + contour plates + head heuristic

        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.model_path = config.get("models", {}).get(
            "detector_weights", "models/best_detector.pt"
        )
        self.thresholds = config.get("models", {}).get(
            "confidence",
            {"rider": 0.35, "helmet": 0.40, "no_helmet": 0.35, "license_plate": 0.35},
        )

        # Primary / combined model
        self.model               = None
        self.names               = {}
        self.has_custom_classes  = False

        # Dual specialist models
        self._plate_model        = None
        self._plate_names        = {}
        self._helmet_model       = None
        self._helmet_names       = {}

        self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Loads the best available model(s)."""
        # 1. Combined custom model
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 100_000:
            print(f"[Detector] Loading combined model from {self.model_path} ...")
            try:
                self.model = YOLO(self.model_path)
                self.names = self.model.names
                self.has_custom_classes = any(
                    any(k in str(v).lower() for k in ["rider", "helmet", "plate", "license"])
                    for v in self.names.values()
                )
                if self.has_custom_classes:
                    print(f"[Detector] Combined model loaded. Classes: {self.names}")
                    return
                else:
                    print("[Detector] Combined model has no custom classes — trying specialist models.")
            except Exception as e:
                print(f"[Detector] Error loading combined model: {e}")

        # 2. Dual specialist models
        plate_ok  = self._load_specialist_model("plate",  _PLATE_MODEL_PATH)
        helmet_ok = self._load_specialist_model("helmet", _HELMET_MODEL_PATH)

        if plate_ok or helmet_ok:
            self.has_custom_classes = True
            print(
                f"[Detector] Specialist models loaded — "
                f"plate={'YES' if plate_ok else 'NO (contour fallback)'}, "
                f"helmet={'YES' if helmet_ok else 'NO (head heuristic)'}"
            )
            return

        # 3. COCO fallback
        print(
            "[Detector] WARNING: No specialist models found. Using generic YOLOv8n (COCO).\n"
            "  Plate detection  -> OpenCV contour heuristic\n"
            "  Helmet detection -> head-crop skin-tone heuristic\n"
            "  Run: python backend/models/download_weights.py  to download real models."
        )
        self._setup_yolo_model()

    def _load_specialist_model(self, kind, path):
        """Loads a single specialist model. Returns True on success."""
        if not (os.path.exists(path) and os.path.getsize(path) > 100_000):
            return False
        try:
            m = YOLO(path)
            if kind == "plate":
                self._plate_model = m
                self._plate_names = m.names
            else:
                self._helmet_model = m
                self._helmet_names = m.names
            print(f"[Detector] {kind.capitalize()} specialist model loaded: {path} | Classes: {m.names}")
            return True
        except Exception as e:
            print(f"[Detector] Failed to load {kind} model ({path}): {e}")
            return False

    def _setup_yolo_model(self):
        """Falls back to standard YOLOv8 COCO model."""
        try:
            self.model = YOLO("yolov8n.pt")
            self.names = self.model.names
            self.has_custom_classes = False
            print("[Detector] YOLOv8n (COCO) fallback model ready.")
        except Exception as e:
            print(f"[Detector] CRITICAL: Cannot load any YOLO model: {e}")
            self.model = None
            self.names = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, img):
        """
        Runs object detection on the input image.

        Returns:
            dict with keys: riders, helmets, no_helmets, plates
            Each value is a list of {"box": np.ndarray[x1,y1,x2,y2], "conf": float}
        """
        empty = {"riders": [], "helmets": [], "no_helmets": [], "plates": []}
        if img is None or img.size == 0:
            return empty

        # Mode A: combined custom model
        if self.has_custom_classes and self.model is not None and \
                self._plate_model is None and self._helmet_model is None:
            return self._detect_combined(img)

        # Mode B: dual specialist models
        if self._plate_model is not None or self._helmet_model is not None:
            return self._detect_dual(img)

        # Mode C: COCO + contour fallback
        return self._detect_coco_fallback(img)

    # ── Mode A: combined model ────────────────────────────────────────────────

    def _detect_combined(self, img):
        results    = self.model(img, verbose=False)[0]
        detections = {"riders": [], "helmets": [], "no_helmets": [], "plates": []}

        for box in results.boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            xyxy       = box.xyxy[0].cpu().numpy().astype(int)
            class_name = self.names.get(cls_id, "").lower()

            if "rider" in class_name and conf >= self.thresholds.get("rider", 0.35):
                detections["riders"].append({"box": xyxy, "conf": conf})
            elif "no" in class_name and "helmet" in class_name and conf >= self.thresholds.get("no_helmet", 0.35):
                detections["no_helmets"].append({"box": xyxy, "conf": conf})
            elif "helmet" in class_name and conf >= self.thresholds.get("helmet", 0.40):
                detections["helmets"].append({"box": xyxy, "conf": conf})
            elif ("plate" in class_name or "license" in class_name) and conf >= self.thresholds.get("license_plate", 0.35):
                detections["plates"].append({"box": xyxy, "conf": conf})

        return detections

    # ── Mode B: dual specialist models ───────────────────────────────────────

    def _detect_dual(self, img):
        detections = {"riders": [], "helmets": [], "no_helmets": [], "plates": []}

        # Helmet / rider model
        if self._helmet_model is not None:
            results = self._helmet_model(img, verbose=False)[0]
            for box in results.boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                xyxy       = box.xyxy[0].cpu().numpy().astype(int)
                class_name = self._helmet_names.get(cls_id, "").lower()

                if "rider" in class_name and conf >= self.thresholds.get("rider", 0.35):
                    detections["riders"].append({"box": xyxy, "conf": conf})
                elif "no" in class_name and "helmet" in class_name and conf >= self.thresholds.get("no_helmet", 0.35):
                    detections["no_helmets"].append({"box": xyxy, "conf": conf})
                elif "helmet" in class_name and conf >= self.thresholds.get("helmet", 0.40):
                    detections["helmets"].append({"box": xyxy, "conf": conf})
                # Some helmet models also detect plates as a side class
                elif ("plate" in class_name or "license" in class_name) and conf >= self.thresholds.get("license_plate", 0.35):
                    detections["plates"].append({"box": xyxy, "conf": conf})

        # Dedicated plate model
        if self._plate_model is not None:
            p_results   = self._plate_model(img, verbose=False)[0]
            plate_th    = self.thresholds.get("license_plate", 0.35)
            for box in p_results.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                if conf >= plate_th:
                    if not self._is_duplicate_box(xyxy, detections["plates"]):
                        detections["plates"].append({"box": xyxy, "conf": conf})

        # If no riders from helmet model, try COCO association
        if not detections["riders"]:
            detections["riders"] = self._coco_rider_detection(img)

        # If still no plates, try contour fallback
        if not detections["plates"]:
            detections["plates"] = self._contour_plate_detection(img)

        return detections

    # ── Mode C: COCO + contour fallback ──────────────────────────────────────

    def _detect_coco_fallback(self, img):
        """
        COCO-mode fallback:
          - Riders      -> person (class 0) + motorcycle (class 3) spatial association
          - Helmets     -> head-region skin-tone heuristic
          - Plates      -> OpenCV contour-based detection
        """
        detections = {"riders": [], "helmets": [], "no_helmets": [], "plates": []}

        if self.model is None:
            return detections

        results  = self.model(img, verbose=False)[0]
        persons, motos = [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)
            if cls_id == 0 and conf >= 0.35:    # person
                persons.append({"box": xyxy, "conf": conf})
            elif cls_id == 3 and conf >= 0.35:  # motorcycle
                motos.append({"box": xyxy, "conf": conf})

        detections["riders"] = self._associate_riders(persons, motos)

        # Head heuristic for each rider
        for rider in detections["riders"]:
            result = self._head_region_heuristic(img, rider["box"])
            if result == "helmet":
                detections["helmets"].append({"box": rider["box"], "conf": 0.5})
            elif result == "no_helmet":
                detections["no_helmets"].append({"box": rider["box"], "conf": 0.5})

        # Contour-based plate detection
        detections["plates"] = self._contour_plate_detection(img)

        return detections

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _coco_rider_detection(self, img):
        """Detects riders via COCO person+motorcycle class association."""
        if self.model is None:
            return []
        results = self.model(img, verbose=False)[0]
        persons, motos = [], []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy().astype(int)
            if cls_id == 0 and conf >= 0.35:
                persons.append({"box": xyxy, "conf": conf})
            elif cls_id == 3 and conf >= 0.35:
                motos.append({"box": xyxy, "conf": conf})
        return self._associate_riders(persons, motos)

    def _associate_riders(self, persons, motos):
        """Associates person detections with overlapping motorcycle detections."""
        riders = []
        for motor in motos:
            m_box      = motor["box"]
            m_x_center = (m_box[0] + m_box[2]) / 2
            m_width    = m_box[2] - m_box[0]
            for person in persons:
                p_box      = person["box"]
                p_x_center = (p_box[0] + p_box[2]) / 2
                if abs(p_x_center - m_x_center) < m_width * 0.7:
                    if p_box[3] > m_box[1] - (m_box[3] - m_box[1]) * 0.3:
                        riders.append({"box": p_box, "conf": person["conf"]})
        return riders

    def _contour_plate_detection(self, img):
        """
        Locates license plate candidates using edge detection + contour geometry.

        Indian license plates are rectangular with aspect ratio ~4-6:1.
        Works on clear, well-lit images without a custom model.
        """
        plates = []
        try:
            h, w     = img.shape[:2]
            gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred  = cv2.bilateralFilter(gray, 11, 17, 17)
            edged    = cv2.Canny(blurred, 30, 200)

            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours     = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (h * w * 0.0005):
                    continue

                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

                if len(approx) == 4:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    if bh == 0:
                        continue
                    aspect   = bw / bh
                    box_area = bw * bh
                    img_area = h * w

                    # Indian plate aspect ratio 2.5-7, area 0.05%-15% of image
                    if 2.5 <= aspect <= 7.0 and (img_area * 0.0005) < box_area < (img_area * 0.15):
                        xyxy = np.array([x, y, x + bw, y + bh])
                        if not self._is_duplicate_box(xyxy, plates):
                            plates.append({"box": xyxy, "conf": 0.45})
        except Exception as e:
            print(f"[Detector] Contour plate detection error: {e}")

        return plates

    def _head_region_heuristic(self, img, rider_box):
        """
        Crops the upper 22% of the rider box and checks skin-tone dominance.
        Returns: "helmet", "no_helmet", or "unknown"
        """
        try:
            x1, y1, x2, y2 = rider_box
            head_h    = max(1, int((y2 - y1) * 0.22))
            head_crop = img[y1:y1 + head_h, x1:x2]
            if head_crop.size == 0:
                return "unknown"
            hsv        = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0,  20,  70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask  = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            if skin_ratio > 0.18:
                return "no_helmet"
            elif skin_ratio < 0.05:
                return "helmet"
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _is_duplicate_box(xyxy, existing, iou_threshold=0.4):
        """Returns True if xyxy overlaps significantly with any box in existing."""
        x1, y1, x2, y2 = xyxy
        area_new = max(0, x2 - x1) * max(0, y2 - y1)
        for item in existing:
            ex  = item["box"]
            ix1 = max(x1, ex[0]); iy1 = max(y1, ex[1])
            ix2 = min(x2, ex[2]); iy2 = min(y2, ex[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_ex = max(0, ex[2] - ex[0]) * max(0, ex[3] - ex[1])
            union   = area_new + area_ex - inter
            if union > 0 and (inter / union) >= iou_threshold:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Triple Riding & Multi-Violation Detection
    # ──────────────────────────────────────────────────────────────────────────

    def detect_triple_riding(self, img, detections):
        """
        Detects instances of Triple Riding (>=3 riders on a two-wheeler)
        by analyzing spatial clustering of riders and motorcycle bounding zones.

        Returns:
            list of dicts: {"box": np.ndarray[x1,y1,x2,y2], "conf": float, "rider_count": int, "rider_boxes": list}
        """
        riders = detections.get("riders", [])
        if len(riders) < 3:
            return []

        triple_riding_clusters = []
        h, w = img.shape[:2]

        # Group riders whose bounding boxes horizontally overlap or are in close proximity
        used_indices = set()
        for i, r1 in enumerate(riders):
            if i in used_indices:
                continue
            box1 = r1["box"]
            cluster = [r1]
            cluster_indices = {i}

            for j, r2 in enumerate(riders):
                if j in cluster_indices or j in used_indices:
                    continue
                box2 = r2["box"]

                # Check horizontal distance between riders
                c1_x = (box1[0] + box1[2]) / 2.0
                c2_x = (box2[0] + box2[2]) / 2.0
                max_width = max(box1[2] - box1[0], box2[2] - box2[0])

                # Vertical alignment check (riders are on roughly same plane)
                y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                min_h = min(box1[3] - box1[1], box2[3] - box2[1])

                if abs(c1_x - c2_x) < max_width * 1.6 and (y_overlap > min_h * 0.3 or abs(box1[1] - box2[1]) < min_h * 0.6):
                    cluster.append(r2)
                    cluster_indices.add(j)

            if len(cluster) >= 3:
                for idx in cluster_indices:
                    used_indices.add(idx)

                all_boxes = [c["box"] for c in cluster]
                min_x = max(0, min(b[0] for b in all_boxes))
                min_y = max(0, min(b[1] for b in all_boxes))
                max_x = min(w, max(b[2] for b in all_boxes))
                max_y = min(h, max(b[3] for b in all_boxes))

                avg_conf = float(np.mean([c["conf"] for c in cluster]))
                triple_riding_clusters.append({
                    "box": np.array([min_x, min_y, max_x, max_y]),
                    "conf": avg_conf,
                    "rider_count": len(cluster),
                    "rider_boxes": all_boxes
                })

        return triple_riding_clusters

    # ──────────────────────────────────────────────────────────────────────────
    # Violation association
    # ──────────────────────────────────────────────────────────────────────────

    def associate_violations(self, detections, speed_kmh=None, speed_limit=60.0):
        """
        Associates detected violations (No Helmet, Triple Riding, Over-Speeding)
        with the closest vehicle license plate.

        Returns:
            list of dicts: each with:
              - "violation_types": list of strings e.g. ["No Helmet", "Triple Riding", "Over-Speeding"]
              - "fine_amount": total fine in INR
              - "no_helmet": no-helmet detection info (or None)
              - "plate": plate detection info
              - "rider": associated rider bounding box
              - "triple_riding": triple riding cluster info (or None)
              - "speed_kmh": recorded speed
              - "severity": "HIGH", "CRITICAL", or "MEDIUM"
        """
        pairs = []
        used_plates = set()

        # Step 1: Detect Triple Riding clusters
        triple_clusters = self.detect_triple_riding(None if not detections.get("riders") else np.zeros((1000, 1000, 3), dtype=np.uint8), detections)

        # Step 2: Handle No-Helmet violations
        for no_helmet in detections.get("no_helmets", []):
            nh_box = no_helmet["box"]
            nh_bottom_center = ((nh_box[0] + nh_box[2]) / 2, nh_box[3])

            best_plate = None
            min_dist   = float("inf")

            for idx, plate in enumerate(detections.get("plates", [])):
                if idx in used_plates:
                    continue
                p_box        = plate["box"]
                p_top_center = ((p_box[0] + p_box[2]) / 2, p_box[1])

                # Plate must be at or below the head detection (with 30-px tolerance)
                if p_top_center[1] > nh_bottom_center[1] - 30:
                    dist = np.hypot(
                        nh_bottom_center[0] - p_top_center[0],
                        nh_bottom_center[1] - p_top_center[1],
                    )
                    if dist < min_dist:
                        min_dist   = dist
                        best_plate = (idx, plate)

            if best_plate is not None:
                idx, plate_info = best_plate
                used_plates.add(idx)

                # Find the rider containing or nearest to this head
                best_rider = None
                r_min_dist = float("inf")
                for rider in detections.get("riders", []):
                    r_box = rider["box"]
                    if (r_box[0] <= nh_box[0] and r_box[1] <= nh_box[1]
                            and r_box[2] >= nh_box[2] and r_box[3] >= nh_box[3]):
                        best_rider = rider
                        break
                    r_center  = ((r_box[0] + r_box[2]) / 2, (r_box[1] + r_box[3]) / 2)
                    nh_center = ((nh_box[0] + nh_box[2]) / 2, (nh_box[1] + nh_box[3]) / 2)
                    dist = np.hypot(r_center[0] - nh_center[0], r_center[1] - nh_center[1])
                    if dist < r_min_dist:
                        r_min_dist = dist
                        best_rider = rider

                # Check if this rider is also part of a triple-riding cluster
                v_types = ["No Helmet"]
                fine = 1000.0
                associated_triple = None

                for tc in triple_clusters:
                    t_box = tc["box"]
                    if (best_rider and t_box[0] <= best_rider["box"][0] and t_box[2] >= best_rider["box"][2]) or \
                       (t_box[0] <= nh_box[0] and t_box[2] >= nh_box[2]):
                        v_types.append(f"Triple Riding ({tc['rider_count']} Riders)")
                        fine += 1000.0
                        associated_triple = tc
                        break

                # Check speed violation
                if speed_kmh is not None and speed_kmh > speed_limit:
                    v_types.append(f"Over-Speeding ({speed_kmh:.1f} km/h)")
                    fine += 2000.0

                severity = "CRITICAL" if len(v_types) > 1 or (speed_kmh and speed_kmh > speed_limit + 20) else "HIGH"

                pairs.append({
                    "violation_types": v_types,
                    "fine_amount": fine,
                    "no_helmet": no_helmet,
                    "plate": plate_info,
                    "rider": best_rider if best_rider else no_helmet,
                    "triple_riding": associated_triple,
                    "speed_kmh": speed_kmh or 0.0,
                    "severity": severity
                })

        # Step 3: Handle standalone Triple-Riding violations if helmet was worn
        for tc in triple_clusters:
            t_box = tc["box"]
            # Check if this cluster was already paired
            already_paired = any(p.get("triple_riding") is not None and np.array_equal(p["triple_riding"]["box"], t_box) for p in pairs)
            if not already_paired:
                # Find nearest unused plate
                best_plate = None
                min_dist = float("inf")
                t_bottom_center = ((t_box[0] + t_box[2]) / 2, t_box[3])

                for idx, plate in enumerate(detections.get("plates", [])):
                    if idx in used_plates:
                        continue
                    p_box = plate["box"]
                    p_top_center = ((p_box[0] + p_box[2]) / 2, p_box[1])
                    dist = np.hypot(t_bottom_center[0] - p_top_center[0], t_bottom_center[1] - p_top_center[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_plate = (idx, plate)

                if best_plate is not None:
                    idx, plate_info = best_plate
                    used_plates.add(idx)
                    v_types = [f"Triple Riding ({tc['rider_count']} Riders)"]
                    fine = 1000.0
                    if speed_kmh is not None and speed_kmh > speed_limit:
                        v_types.append(f"Over-Speeding ({speed_kmh:.1f} km/h)")
                        fine += 2000.0

                    pairs.append({
                        "violation_types": v_types,
                        "fine_amount": fine,
                        "no_helmet": None,
                        "plate": plate_info,
                        "rider": {"box": t_box, "conf": tc["conf"]},
                        "triple_riding": tc,
                        "speed_kmh": speed_kmh or 0.0,
                        "severity": "HIGH"
                    })

        return pairs

    # ──────────────────────────────────────────────────────────────────────────
    # Annotation
    # ──────────────────────────────────────────────────────────────────────────

    def draw_annotations(self, img, detections, violations, tracked_vehicles=None, speed_limit=60.0):
        """
        Draws HUD bounding boxes and labels on the image for visualisation.
        - Yellow/Cyan -> Rider bounding box
        - Green       -> Helmet compliant
        - Red         -> No-Helmet violation + associated plate
        - Purple      -> Triple-Riding cluster
        - Crimson     -> Speed violation alert HUD
        - Orange/Blue -> Standard detected plate
        """
        annotated = img.copy()
        violating_plate_boxes = [v["plate"]["box"] for v in violations if "plate" in v and v["plate"] is not None]

        # Draw Riders
        for r in detections.get("riders", []):
            box = r["box"]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 200, 0), 2)
            cv2.putText(annotated, f"Rider {r['conf']:.2f}", (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # Draw Helmets
        for h in detections.get("helmets", []):
            box = h["box"]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(annotated, "Helmet", (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Draw No-Helmets
        for nh in detections.get("no_helmets", []):
            box = nh["box"]
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(annotated, "NO HELMET", (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw Triple Riding overlays
        for v in violations:
            if v.get("triple_riding") is not None:
                t_box = v["triple_riding"]["box"]
                count = v["triple_riding"].get("rider_count", 3)
                cv2.rectangle(annotated, (t_box[0], t_box[1]), (t_box[2], t_box[3]), (255, 0, 180), 3)
                cv2.putText(annotated, f"TRIPLE RIDING ({count} Riders)", (t_box[0], max(20, t_box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 180), 2)

        # Draw Plates
        for p in detections.get("plates", []):
            box = p["box"]
            is_violating = any(np.array_equal(box, vb) for vb in violating_plate_boxes)
            color = (0, 0, 255) if is_violating else (255, 100, 0)
            label = "Plate (VIOLATION)" if is_violating else f"Plate {p['conf']:.2f}"
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(annotated, label, (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw Tracked vehicle overlays and speeds if provided
        if tracked_vehicles is not None:
            for track_id, info in tracked_vehicles.items():
                box = info.get("box")
                speed = info.get("speed", 0.0)
                if box is not None:
                    is_overspeed = speed > speed_limit
                    spd_color = (0, 50, 255) if is_overspeed else (0, 255, 180)
                    spd_label = f"ID:{track_id} | {speed:.0f} km/h {'[SPEED ALERT]' if is_overspeed else ''}"
                    cv2.putText(annotated, spd_label, (box[0], min(annotated.shape[0] - 10, box[3] + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, spd_color, 2)

        return annotated
