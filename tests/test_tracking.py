import unittest
import time
from backend.tracker import VehicleTracker

class TestVehicleTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = VehicleTracker(max_disappeared=5, cooldown_seconds=2)

    def test_tracking_registration(self):
        # Frame 1 with two bounding boxes
        rects = [[10, 10, 50, 50], [100, 100, 150, 150]]
        assigned = self.tracker.update(rects)
        self.assertEqual(len(assigned), 2)
        self.assertIn(1, assigned)
        self.assertIn(2, assigned)

    def test_speed_estimation(self):
        rects_f1 = [[10, 10, 50, 50]]
        self.tracker.update(rects_f1)
        time.sleep(0.1)
        
        rects_f2 = [[50, 50, 90, 90]]
        self.tracker.update(rects_f2)
        
        speed = self.tracker.estimate_speed(1)
        self.assertIsInstance(speed, float)
        self.assertGreaterEqual(speed, 0.0)

    def test_duplicate_plate_suppression(self):
        plate = "MH12DE5678"
        self.assertFalse(self.tracker.is_duplicate_plate(plate))
        
        self.tracker.mark_plate_ticketed(plate)
        self.assertTrue(self.tracker.is_duplicate_plate(plate))
        
        # Wait out cooldown
        time.sleep(2.1)
        self.assertFalse(self.tracker.is_duplicate_plate(plate))

if __name__ == "__main__":
    unittest.main()
