import unittest
from backend.stream_handler import RTSPStreamManager, RTSPCameraStream

class TestRTSPStreamHandler(unittest.TestCase):
    def setUp(self):
        self.manager = RTSPStreamManager()

    def test_add_and_status(self):
        cam = self.manager.add_stream("CAM_TEST", "test_assets/traffic_sample.png")
        self.assertIsNotNone(cam)
        status = self.manager.get_stream_status()
        self.assertIn("CAM_TEST", status)
        self.manager.stop_all()

if __name__ == "__main__":
    unittest.main()
