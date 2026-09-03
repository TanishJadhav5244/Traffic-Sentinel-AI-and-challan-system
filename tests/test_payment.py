import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db_helper import ViolationDatabase
from backend.challan_payment import EChallanPaymentGateway, DemoPaymentGateway, RazorpayGateway

class TestPaymentGateway(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_violations.csv")
        self.crop_dir = os.path.join(self.temp_dir, "crops")
        self.challan_dir = os.path.join(self.temp_dir, "challans")
        
        self.db = ViolationDatabase(csv_log_path=self.csv_path, crop_dir=self.crop_dir)
        self.gateway = EChallanPaymentGateway(self.db, output_dir=self.challan_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_process_payment_flow(self):
        # Insert test violation
        record = self.db.add_violation(
            plate_text="MH12DE5678",
            ocr_confidence=0.92,
            helmet_status="No-Helmet",
            plate_crop_path="",
            rider_crop_path="",
            owner_name="Rajesh Kumar",
            vehicle_model="TVS Jupiter 125",
            challan_amount=1000.0,
            challan_path="",
            night_mode=False
        )
        v_id = record["violation_id"] if isinstance(record, dict) else record
        
        res = self.gateway.process_payment(v_id, payment_method="UPI_PAYTM")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["payment_status"], "PAID")
        self.assertTrue(os.path.exists(res["receipt_path"]))

    def test_razorpay_gateway_fallback(self):
        rzp = RazorpayGateway({"razorpay_key_id": "", "razorpay_key_secret": ""})
        res = rzp.initiate_payment("TEST1234", 1000.0, "UPI_ONLINE")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("RazorpayGateway", res["gateway"])

if __name__ == "__main__":
    unittest.main()

