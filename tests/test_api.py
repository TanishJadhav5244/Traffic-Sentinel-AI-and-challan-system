import unittest
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.api import app

class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_health_check(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["system"], "Traffic Sentinel AI")
        self.assertEqual(data["version"], "2.0.0")

    def test_get_violations_endpoint(self):
        response = self.client.get("/api/v1/violations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("violations", data)
        self.assertIn("total_records", data)

    def test_get_analytics_summary_endpoint(self):
        response = self.client.get("/api/v1/analytics/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary", data)
        self.assertIn("hourly_distribution", data)

    def test_scan_image_endpoint(self):
        # Create a dummy test image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, encoded_img = cv2.imencode(".png", img)
        
        response = self.client.post(
            "/api/v1/scan/image",
            files={"file": ("test.png", encoded_img.tobytes(), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("violations", data)

if __name__ == "__main__":
    unittest.main()
