import cv2
import time
import threading
from queue import Queue

class RTSPCameraStream:
    """
    Multi-threaded RTSP IP Camera stream handler.
    Continuously captures frames in a dedicated background thread with
    ring buffer management to prevent frame lag and auto-reconnect on disconnects.
    """
    def __init__(self, stream_url, camera_id="CAM_01", queue_size=128):
        self.stream_url = stream_url
        self.camera_id = camera_id
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.connected = False
        self.cap = None
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.thread = None

    def start(self):
        """Starts background frame reader thread."""
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Background worker thread capturing frames from RTSP stream."""
        while not self.stopped:
            if not self.connected or self.cap is None or not self.cap.isOpened():
                print(f"[RTSP {self.camera_id}] Connecting to stream: {self.stream_url}")
                self.cap = cv2.VideoCapture(self.stream_url)
                if self.cap.isOpened():
                    self.connected = True
                    print(f"[RTSP {self.camera_id}] Stream connected successfully.")
                else:
                    self.connected = False
                    time.sleep(2.0) # Wait before retry
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.connected = False
                time.sleep(0.5)
                continue

            # Calculate FPS
            now = time.time()
            dt = now - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_frame_time = now

            # Maintain queue size: drop oldest frame if full to prevent latency lag
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass

            self.queue.put((now, frame))

        if self.cap is not None:
            self.cap.release()

    def read(self):
        """Reads the latest frame from the queue."""
        if not self.queue.empty():
            return self.queue.get()
        return None, None

    def stop(self):
        """Stops stream capture thread."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)

class RTSPStreamManager:
    """Manages multiple concurrent RTSP camera streams for traffic monitoring grid."""
    def __init__(self):
        self.streams = {}

    def add_stream(self, camera_id, stream_url):
        if camera_id in self.streams:
            self.streams[camera_id].stop()
        cam = RTSPCameraStream(stream_url, camera_id=camera_id)
        cam.start()
        self.streams[camera_id] = cam
        return cam

    def get_latest_frame(self, camera_id):
        if camera_id in self.streams:
            return self.streams[camera_id].read()
        return None, None

    def get_stream_status(self):
        status = {}
        for cid, cam in self.streams.items():
            status[cid] = {
                "connected": cam.connected,
                "fps": round(cam.fps, 1),
                "url": cam.stream_url
            }
        return status

    def stop_all(self):
        for cam in self.streams.values():
            cam.stop()
        self.streams.clear()
