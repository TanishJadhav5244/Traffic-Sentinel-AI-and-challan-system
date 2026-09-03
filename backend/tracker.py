import time
import math
import numpy as np

class VehicleTracker:
    """
    Vehicle Tracker for frame-to-frame association, track ID persistence,
    speed estimation (px/s converted to km/h using spatial calibration),
    and duplicate license plate suppression over a rolling time window.
    """
    def __init__(self, max_disappeared=30, min_distance_threshold=100.0, cooldown_seconds=300, pixels_per_meter=20.0):
        self.next_object_id = 1
        self.objects = {}        # track_id -> centroid (x, y)
        self.disappeared = {}      # track_id -> disappeared frame count
        self.track_history = {}  # track_id -> list of (timestamp, (x, y))
        
        self.max_disappeared = max_disappeared
        self.min_distance_threshold = min_distance_threshold
        self.cooldown_seconds = cooldown_seconds
        self.pixels_per_meter = pixels_per_meter
        
        # Suppress duplicate license plate issuance within cooldown window
        self.ticketed_plates = {} # plate_number -> last_ticketed_timestamp

    def register(self, centroid):
        """Registers a new vehicle object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.track_history[self.next_object_id] = [(time.time(), centroid)]
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """Deregisters a lost vehicle object."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.track_history:
            del self.track_history[object_id]

    def update(self, rects):
        """
        Updates tracked object positions given a list of bounding boxes [x1, y1, x2, y2].
        Returns a dict mapping track_id -> box.
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            assigned_boxes = {}
            for i in range(len(input_centroids)):
                tid = self.register(input_centroids[i])
                assigned_boxes[tid] = rects[i]
            return assigned_boxes

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Compute distance matrix between existing centroids and input centroids
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        assigned_boxes = {}
        curr_time = time.time()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.min_distance_threshold:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            self.track_history[object_id].append((curr_time, input_centroids[col]))
            
            # Keep history capped to last 30 frames
            if len(self.track_history[object_id]) > 30:
                self.track_history[object_id].pop(0)

            assigned_boxes[object_id] = rects[col]
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            tid = self.register(input_centroids[col])
            assigned_boxes[tid] = rects[col]

        return assigned_boxes

    def estimate_speed(self, track_id):
        """
        Estimates vehicle speed in km/h based on recent centroid movement.
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0.0

        history = self.track_history[track_id]
        t1, (x1, y1) = history[0]
        t2, (x2, y2) = history[-1]

        dt = t2 - t1
        if dt <= 0:
            return 0.0

        pixel_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        meters = pixel_dist / max(self.pixels_per_meter, 1.0)
        mps = meters / dt
        kmh = mps * 3.6
        return round(kmh, 1)

    def is_speed_violation(self, track_id, speed_limit=60.0):
        """Returns True if the estimated speed exceeds the speed limit."""
        speed = self.estimate_speed(track_id)
        return speed > speed_limit, speed

    def get_trajectory(self, track_id):
        """Returns the list of (x, y) centroid points for the track trajectory."""
        if track_id in self.track_history:
            return [pt for _, pt in self.track_history[track_id]]
        return []

    def get_all_active_tracks(self, assigned_boxes, speed_limit=60.0):
        """
        Returns a dictionary of all active tracks with their bounding box,
        centroid, estimated speed, speed alert status, and trajectory history.
        """
        tracks = {}
        for track_id, box in assigned_boxes.items():
            speed = self.estimate_speed(track_id)
            tracks[track_id] = {
                "box": box,
                "centroid": self.objects.get(track_id, (0, 0)),
                "speed": speed,
                "is_speed_violation": speed > speed_limit,
                "trajectory": self.get_trajectory(track_id)
            }
        return tracks

    def is_duplicate_plate(self, plate_number):
        """
        Checks whether a plate was ticketed recently within the cooldown window.
        """
        if not plate_number or len(plate_number.strip()) < 4:
            return False

        clean_plate = plate_number.upper().replace(" ", "")
        now = time.time()
        
        if clean_plate in self.ticketed_plates:
            last_time = self.ticketed_plates[clean_plate]
            if now - last_time < self.cooldown_seconds:
                return True

        return False

    def mark_plate_ticketed(self, plate_number):
        """Marks a plate as ticketed at current timestamp."""
        if plate_number and len(plate_number.strip()) >= 4:
            clean_plate = plate_number.upper().replace(" ", "")
            self.ticketed_plates[clean_plate] = time.time()

