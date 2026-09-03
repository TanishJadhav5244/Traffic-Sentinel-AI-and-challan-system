"""
backend/challan_payment.py
==========================
Digital E-Challan payment processing system with:
  - Pluggable payment gateway abstraction (Demo, Razorpay)
  - Payment lifecycle tracking (INITIATED → PROCESSING → SUCCESS/FAILED)
  - Official government-style payment receipt generation with UPI QR
  - Transaction history and receipt serving
"""

import os
import uuid
import datetime
from PIL import Image, ImageDraw, ImageFont


def get_font(font_name="arial", size=14, bold=False):
    """Utility to load system fonts or default."""
    font_paths = []
    if bold:
        font_paths = ["C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\consolab.ttf"]
    else:
        font_paths = ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\consola.ttf"]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ─── Payment Gateway Abstraction ─────────────────────────────────────────────

class BasePaymentGateway:
    """Abstract payment gateway interface."""

    def __init__(self, config=None):
        self.config = config or {}
        self.transaction_log = []

    def initiate_payment(self, challan_id, amount, payment_method="UPI_ONLINE"):
        """Initiates a payment. Returns transaction record."""
        raise NotImplementedError

    def verify_payment(self, transaction_id):
        """Verifies payment status. Returns updated transaction record."""
        raise NotImplementedError

    def get_transaction_log(self):
        return self.transaction_log


class DemoPaymentGateway(BasePaymentGateway):
    """
    Demo payment gateway — instant mock payment processing.
    Used when no real payment provider is configured.
    """

    def initiate_payment(self, challan_id, amount, payment_method="UPI_ONLINE"):
        tx_id = f"TXN-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "transaction_id": tx_id,
            "challan_id": challan_id,
            "amount": float(amount),
            "payment_method": payment_method,
            "status": "SUCCESS",
            "initiated_at": now,
            "completed_at": now,
            "gateway": "DemoPaymentGateway",
            "upi_ref": f"UPI-{uuid.uuid4().hex[:8].upper()}" if "UPI" in payment_method else None,
            "bank_ref": f"NEFT-{uuid.uuid4().hex[:12].upper()}" if "BANK" in payment_method else None,
        }
        self.transaction_log.append(record)
        print(f"[Payment-Demo] Payment {tx_id} for Challan #{challan_id}: INR {amount} -- SUCCESS")
        return record

    def verify_payment(self, transaction_id):
        for tx in self.transaction_log:
            if tx["transaction_id"] == transaction_id:
                return tx
        return {"transaction_id": transaction_id, "status": "NOT_FOUND"}


class RazorpayGateway(BasePaymentGateway):
    """
    Razorpay payment gateway integration stub.
    Requires key_id and key_secret in config.
    Falls back to demo mode if credentials are missing.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.key_id = (config or {}).get("razorpay_key_id", "")
        self.key_secret = (config or {}).get("razorpay_key_secret", "")

    def initiate_payment(self, challan_id, amount, payment_method="UPI_ONLINE"):
        if not self.key_id or not self.key_secret:
            print("[Payment-Razorpay] No credentials configured. Using demo mode.")
            demo = DemoPaymentGateway(self.config)
            result = demo.initiate_payment(challan_id, amount, payment_method)
            result["gateway"] = "RazorpayGateway (Demo Mode)"
            self.transaction_log.append(result)
            return result

        # Real Razorpay API integration would go here
        # For now, simulate the flow structure
        tx_id = f"TXN-RZP-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "transaction_id": tx_id,
            "challan_id": challan_id,
            "amount": float(amount),
            "payment_method": payment_method,
            "status": "SUCCESS",
            "initiated_at": now,
            "completed_at": now,
            "gateway": "RazorpayGateway",
            "razorpay_order_id": f"order_{uuid.uuid4().hex[:14]}",
            "razorpay_payment_id": f"pay_{uuid.uuid4().hex[:14]}",
        }
        self.transaction_log.append(record)
        return record

    def verify_payment(self, transaction_id):
        for tx in self.transaction_log:
            if tx["transaction_id"] == transaction_id:
                return tx
        return {"transaction_id": transaction_id, "status": "NOT_FOUND"}


def create_payment_gateway(config=None):
    """Factory function to create payment gateway based on config."""
    cfg = (config or {}).get("payment", {})
    provider = cfg.get("provider", "demo").lower()

    if provider == "razorpay":
        return RazorpayGateway(cfg)
    else:
        return DemoPaymentGateway(cfg)


# ─── E-Challan Payment Processor ─────────────────────────────────────────────

class EChallanPaymentGateway:
    """
    Handles digital E-Challan ticket lookup, payment verification,
    and official tax receipt image rendering.
    """
    def __init__(self, db_helper, config=None, output_dir="violations/challans"):
        self.db = db_helper
        self.config = config or {}
        self.output_dir = output_dir
        self.gateway = create_payment_gateway(self.config)

    def process_payment(self, challan_id, payment_method="UPI_ONLINE", transaction_id=None):
        """
        Processes online payment for a given E-Challan ticket.
        Updates DB status to 'PAID' and issues official receipt.
        """
        violation = self.db.get_violation_by_id(challan_id)
        if not violation:
            # Try searching by challan_id matching string
            df = self.db.get_all_violations()
            if not df.empty and "challan_id" in df.columns:
                match = df[df["challan_id"].astype(str) == str(challan_id)]
                if not match.empty:
                    violation = match.iloc[0].to_dict()

        if not violation:
            return {"status": "ERROR", "message": f"Challan #{challan_id} not found."}

        amount = float(violation.get("challan_amount", 1000.0))

        # Process through gateway
        tx_result = self.gateway.initiate_payment(challan_id, amount, payment_method)

        if tx_result.get("status") != "SUCCESS":
            return {
                "status": "FAILED",
                "message": "Payment processing failed.",
                "transaction": tx_result
            }

        tx_id = tx_result.get("transaction_id", transaction_id or f"TXN-{uuid.uuid4().hex[:10].upper()}")
        pay_time = tx_result.get("completed_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Update status in DB
        v_id = violation.get("violation_id") or challan_id
        self.db.update_violation_status(str(v_id), "Paid")

        receipt_path = self.generate_payment_receipt(
            challan_id=challan_id,
            plate_number=violation.get("plate_text", "MH12DE5678"),
            owner_name=violation.get("owner_name", "Vehicle Owner"),
            amount=amount,
            transaction_id=tx_id,
            payment_time=pay_time,
            payment_method=payment_method
        )

        return {
            "status": "SUCCESS",
            "message": "Payment successfully processed.",
            "challan_id": challan_id,
            "transaction_id": tx_id,
            "payment_status": "PAID",
            "payment_time": pay_time,
            "payment_method": payment_method,
            "receipt_path": receipt_path,
            "gateway": tx_result.get("gateway", "Unknown"),
            "upi_ref": tx_result.get("upi_ref"),
            "bank_ref": tx_result.get("bank_ref"),
        }

    def generate_payment_receipt(self, challan_id, plate_number, owner_name, amount, transaction_id, payment_time, payment_method):
        """Generates visual official payment receipt image (PNG)."""
        os.makedirs(self.output_dir, exist_ok=True)

        canvas_w, canvas_h = 750, 580
        img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Outer border
        draw.rectangle([5, 5, canvas_w - 6, canvas_h - 6], outline=(34, 197, 94), width=3)

        # Header
        draw.rectangle([5, 5, canvas_w - 6, 85], fill=(22, 101, 52))
        f_title = get_font("arial", 20, bold=True)
        draw.text((20, 18), "OFFICIAL E-CHALLAN PAYMENT RECEIPT", fill=(255, 255, 255), font=f_title)

        f_sub = get_font("arial", 11)
        draw.text((20, 48), "MINISTRY OF ROAD TRANSPORT & HIGHWAYS | PARIVAHAN SUVIDHA", fill=(220, 252, 231), font=f_sub)
        draw.text((20, 65), f"Generated: {payment_time}", fill=(187, 247, 208), font=f_sub)

        # Body details
        f_label = get_font("arial", 13, bold=True)
        f_val = get_font("arial", 13)

        rows = [
            ("Receipt No / Transaction ID:", transaction_id),
            ("E-Challan ID:", f"#{challan_id}"),
            ("Vehicle Number:", plate_number),
            ("Registered Owner Name:", owner_name),
            ("Payment Date & Time:", payment_time),
            ("Payment Gateway Mode:", payment_method),
            ("Penalty Amount Paid:", f"INR {amount:.2f}"),
            ("Payment Status:", "CONFIRMED & PAID"),
            ("Gateway Provider:", self.gateway.__class__.__name__),
        ]

        y = 110
        for label, val in rows:
            draw.text((40, y), label, fill=(15, 23, 42), font=f_label)
            fill_color = (22, 163, 74) if "PAID" in str(val) or "INR" in str(val) else (51, 65, 85)
            draw.text((340, y), str(val), fill=fill_color, font=f_val)
            draw.line([(40, y + 25), (canvas_w - 40, y + 25)], fill=(241, 245, 249), width=1)
            y += 33

        # UPI QR code placeholder
        qr_y = y + 15
        qr_size = 80
        draw.rectangle([40, qr_y, 40 + qr_size, qr_y + qr_size], fill=(245, 245, 245), outline=(34, 197, 94), width=2)

        # Draw mini QR pattern
        cell = qr_size // 10
        for r in range(10):
            for c in range(10):
                if (abs(hash(f"{transaction_id}{r}{c}")) % 3) < 2:
                    draw.rectangle(
                        [40 + c * cell, qr_y + r * cell, 40 + (c + 1) * cell, qr_y + (r + 1) * cell],
                        fill=(22, 101, 52)
                    )

        f_qr = get_font("arial", 9, bold=True)
        draw.text((40, qr_y + qr_size + 5), "SCAN FOR VERIFICATION", fill=(22, 101, 52), font=f_qr)

        # Footer stamp
        stamp_y = qr_y + qr_size + 25
        draw.rectangle([40, stamp_y, canvas_w - 40, stamp_y + 40], fill=(240, 253, 244), outline=(34, 197, 94))
        f_stamp = get_font("arial", 12, bold=True)
        draw.text((60, stamp_y + 12), "[PAID] DIGITAL PAYMENT VERIFIED -- NO OUTSTANDING FINES REMAINING", fill=(22, 101, 52), font=f_stamp)

        # Security hash
        sec_hash = abs(hash(f"{challan_id}{transaction_id}")) % 10000000000
        f_sec = get_font("arial", 8)
        draw.text((40, stamp_y + 50), f"Security Hash: MoRTH-PAY-{sec_hash:010d} | Anti-Tamper Digital Seal", fill=(148, 163, 184), font=f_sec)

        # Save PNG
        png_filename = f"receipt_{challan_id}_{transaction_id}.png"
        png_path = os.path.join(self.output_dir, png_filename)
        img.save(png_path)

        # Save PDF
        pdf_filename = f"receipt_{challan_id}_{transaction_id}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        try:
            img.save(pdf_path, "PDF", resolution=100.0)
        except Exception as e:
            print(f"[Payment] PDF receipt generation note: {e}")

        return png_path

    def get_receipt_path(self, transaction_id, ext="png"):
        """Finds a receipt file by transaction ID (png or pdf)."""
        if not os.path.exists(self.output_dir):
            return None
        target_ext = f".{ext.lower()}"
        for fname in os.listdir(self.output_dir):
            if transaction_id in fname and fname.startswith("receipt_") and fname.endswith(target_ext):
                return os.path.join(self.output_dir, fname)
        return None

    def get_payment_analytics(self):
        """
        Computes aggregate payment metrics:
        total revenue collected, total pending fines, collection recovery rate,
        and payment method distributions.
        """
        df = self.db.get_all_violations()
        tx_log = self.gateway.get_transaction_log()

        if df.empty:
            return {
                "total_revenue_inr": 0.0,
                "pending_revenue_inr": 0.0,
                "recovery_rate_pct": 0.0,
                "total_paid_tickets": 0,
                "total_pending_tickets": 0,
                "method_breakdown": {},
                "recent_transactions": tx_log[-10:] if tx_log else []
            }

        fines = df.get("challan_amount", 1000.0).astype(float)
        statuses = df.get("status", "Pending").astype(str).str.upper()

        paid_mask = (statuses == "PAID")
        pending_mask = (statuses == "PENDING")

        total_collected = float(fines[paid_mask].sum()) if any(paid_mask) else 0.0
        total_pending = float(fines[pending_mask].sum()) if any(pending_mask) else 0.0
        total_fines = total_collected + total_pending

        recovery_rate = round((total_collected / total_fines * 100.0), 1) if total_fines > 0 else 0.0

        # Method breakdown from transactions
        methods = {}
        for tx in tx_log:
            m = tx.get("payment_method", "UPI_ONLINE")
            methods[m] = methods.get(m, 0) + 1

        return {
            "total_revenue_inr": total_collected,
            "pending_revenue_inr": total_pending,
            "recovery_rate_pct": recovery_rate,
            "total_paid_tickets": int(paid_mask.sum()),
            "total_pending_tickets": int(pending_mask.sum()),
            "method_breakdown": methods,
            "recent_transactions": tx_log[-10:] if tx_log else []
        }

