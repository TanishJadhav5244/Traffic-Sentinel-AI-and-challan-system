"""
tests/test_pipeline.py
======================
Unit and integration tests for Traffic Sentinel AI modules:
- image_enhancer
- ocr_engine
- detector
- rto_helper
- challan_generator
- db_helper
"""

import os
import cv2
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd

from backend.image_enhancer import is_low_light, get_luminance, enhance
from backend.ocr_engine import LicensePlateOCR
from backend.detector import VehicleHelmetDetector
from backend.rto_helper import query_rto
from backend.challan_generator import generate_challan_ticket
from backend.db_helper import ViolationDatabase


class TestImageEnhancer(unittest.TestCase):
    def setUp(self):
        # Create dummy dark and bright images
        self.bright_img = np.full((100, 200, 3), 200, dtype=np.uint8)
        self.dark_img = np.full((100, 200, 3), 25, dtype=np.uint8)

    def test_luminance_calculation(self):
        lum_bright = get_luminance(self.bright_img)
        lum_dark = get_luminance(self.dark_img)
        self.assertGreater(lum_bright, 150.0)
        self.assertLess(lum_dark, 50.0)

    def test_low_light_detection(self):
        self.assertFalse(is_low_light(self.bright_img))
        self.assertTrue(is_low_light(self.dark_img))

    def test_enhancement_pipeline(self):
        # Test default dark image enhancement
        result = enhance(self.dark_img)
        self.assertTrue(result["was_low_light"])
        self.assertGreater(result["luminance_after"], result["luminance_before"])
        self.assertIn("CLAHE", result["stages_applied"])

    def test_enhancement_forced(self):
        result = enhance(self.bright_img, force=True)
        self.assertTrue(len(result["stages_applied"]) > 0)


class TestOCREngine(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ocr": {
                "default_engine": "easyocr",
                "preprocessing": {"resize_factor": 2.0, "bilateral_filter": True, "adaptive_threshold": True, "deskew": True},
                "enhancer": {"auto_low_light": True, "luminance_threshold": 85, "force_enhancement": False}
            }
        }
        self.ocr = LicensePlateOCR(self.config)
        self.sample_img = np.full((50, 150, 3), 180, dtype=np.uint8)

    def test_clean_plate_text(self):
        self.assertEqual(self.ocr.clean_plate_text("MH-12 AB 1234"), "MH12AB1234")
        self.assertEqual(self.ocr.clean_plate_text("mh12  de 5678!"), "MH12DE5678")
        # Test 10-char heuristic correction (e.g., 'O' -> '0' in district code)
        self.assertEqual(self.ocr.clean_plate_text("MHO2AB1234"), "MH02AB1234")

    def test_preprocessing(self):
        preprocessed = self.ocr.preprocess_image(self.sample_img)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(len(preprocessed.shape), 2)  # Should be grayscale/binary

    def test_recognize_schema(self):
        res = self.ocr.recognize(self.sample_img, engine="easyocr")
        self.assertIn("raw_text", res)
        self.assertIn("cleaned_text", res)
        self.assertIn("confidence", res)
        self.assertIn("enhanced_img", res)
        self.assertIn("was_low_light", res)


class TestVehicleHelmetDetector(unittest.TestCase):
    def setUp(self):
        self.config = {
            "models": {
                "detector_weights": "models/non_existent.pt",
                "confidence": {"rider": 0.35, "helmet": 0.40, "no_helmet": 0.35, "license_plate": 0.40}
            }
        }
        self.detector = VehicleHelmetDetector(self.config)

    def test_detector_initialization(self):
        self.assertFalse(self.detector.has_custom_classes)
        self.assertIsNotNone(self.detector.model)

    def test_detect_blank_image_returns_empty_no_mock(self):
        # Blank image must return 0 riders/helmets/plates — NO fabricated mock detections
        dummy_frame = np.full((300, 400, 3), 128, dtype=np.uint8)
        detections = self.detector.detect(dummy_frame)
        self.assertEqual(len(detections["riders"]), 0)
        self.assertEqual(len(detections["helmets"]), 0)
        self.assertEqual(len(detections["no_helmets"]), 0)
        self.assertEqual(len(detections["plates"]), 0)


class TestRTOHelper(unittest.TestCase):
    def test_query_rto_unconfigured(self):
        # Default config without API key returns clearly labeled unavailable data
        info = query_rto("MH12DE5678", config={})
        self.assertEqual(info["owner_name"], "[No RTO API Configured]")
        self.assertEqual(info["status"], "Unconfigured")
        self.assertEqual(info["lookup_status"], "api_not_configured")

    def test_query_rto_demo_opt_in(self):
        # Explicit opt-in demo mode returns labeled demo records
        config = {"rto": {"demo_fallback": True}}
        info = query_rto("MH12DE5678", config=config)
        self.assertIn("[DEMO]", info["owner_name"])
        self.assertEqual(info["api_source"], "Demo Fallback Registry")


class TestChallanGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_generate_challan_ticket(self):
        plate_crop = np.full((40, 120, 3), 100, dtype=np.uint8)
        rider_crop = np.full((100, 100, 3), 150, dtype=np.uint8)
        rto_info = query_rto("MH12AB1234")
        
        ticket_path = generate_challan_ticket(
            violation_id="TEST001",
            timestamp="2026-08-15 12:00:00",
            plate_text="MH12AB1234",
            rto_info=rto_info,
            plate_crop=plate_crop,
            rider_crop=rider_crop,
            output_dir=self.temp_dir
        )
        
        self.assertTrue(os.path.exists(ticket_path))
        self.assertTrue(ticket_path.endswith(".png"))


class TestViolationDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_log.csv")
        self.crop_dir = os.path.join(self.temp_dir, "crops")
        self.db = ViolationDatabase(csv_log_path=self.csv_path, crop_dir=self.crop_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_log_violation_and_retrieval(self):
        plate_crop = np.full((30, 90, 3), 200, dtype=np.uint8)
        rider_crop = np.full((80, 80, 3), 180, dtype=np.uint8)

        record = self.db.log_violation(
            frame_timestamp="00:00:05",
            plate_crop=plate_crop,
            rider_crop=rider_crop,
            plate_text="DL3CAY1111",
            ocr_conf=0.92,
            helmet_status="no-helmet",
            night_mode=True
        )

        self.assertEqual(record["plate_text"], "DL3CAY1111")
        self.assertTrue(record["night_mode"])

        df = self.db.get_all_violations()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["plate_text"], "DL3CAY1111")

    def test_analytics_generation(self):
        plate_crop = np.full((30, 90, 3), 200, dtype=np.uint8)
        self.db.log_violation("00:00:01", plate_crop, plate_crop, "MH12DE5678", 0.95)
        self.db.log_violation("00:00:02", plate_crop, plate_crop, "MH12AB1234", 0.88)
        
        analytics = self.db.get_analytics_data()
        self.assertEqual(analytics["total"], 2)
        self.assertEqual(analytics["total_fines"], 2000.0)
        self.assertFalse(analytics["state_counts"].empty)

    def test_update_violation_status(self):
        plate_crop = np.full((30, 90, 3), 200, dtype=np.uint8)
        record = self.db.log_violation("00:00:01", plate_crop, plate_crop, "KA01AB1234", 0.90)
        v_id = record["violation_id"]
        
        self.assertEqual(record["status"], "Pending")
        updated = self.db.update_violation_status(v_id, "Paid")
        self.assertTrue(updated)
        
        df = self.db.get_all_violations()
        self.assertEqual(df.iloc[0]["status"], "Paid")

    def test_generate_challan_pdf(self):
        from backend.challan_generator import generate_challan_pdf
        plate_crop = np.full((30, 90, 3), 200, dtype=np.uint8)
        rider_crop = np.full((80, 80, 3), 180, dtype=np.uint8)
        rto_info = {"owner_name": "Test Owner", "vehicle_model": "Test Bike"}
        
        pdf_path = generate_challan_pdf("TEST1234", "2026-08-23 12:00:00", "MH12AB1234", rto_info, plate_crop, rider_crop, output_dir=self.temp_dir)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertTrue(pdf_path.endswith(".pdf"))


if __name__ == "__main__":
    unittest.main()

