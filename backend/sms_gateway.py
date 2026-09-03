"""
backend/sms_gateway.py
======================
Pluggable SMS gateway abstraction for sending E-Challan violation
notifications via SMS. Supports multiple providers:
  - DemoSMSGateway   — logs to console/file (default, no credentials needed)
  - TwilioSMSGateway  — real Twilio API integration
  - Fast2SMSGateway   — Indian bulk SMS provider (Fast2SMS DLT route)
"""

import os
import json
import datetime
import urllib.request
import urllib.parse


class BaseSMSGateway:
    """Abstract base class for SMS gateway providers."""

    def __init__(self, config=None):
        self.config = config or {}
        self.delivery_log = []

    def send(self, phone_number, message, metadata=None):
        """
        Sends an SMS message. Returns a delivery record dict.

        Args:
            phone_number (str): Recipient phone number (with country code).
            message (str): SMS body text.
            metadata (dict): Optional extra fields (challan_id, plate, etc.).

        Returns:
            dict: Delivery record with status, timestamp, provider info.
        """
        raise NotImplementedError("Subclasses must implement send()")

    def get_delivery_log(self):
        """Returns the full delivery log."""
        return self.delivery_log

    def _create_record(self, phone_number, message, status, provider, metadata=None):
        record = {
            "type": "SMS",
            "provider": provider,
            "recipient": phone_number,
            "message": message[:160],  # SMS character limit
            "status": status,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **(metadata or {})
        }
        self.delivery_log.append(record)
        return record


class DemoSMSGateway(BaseSMSGateway):
    """
    Demo SMS gateway that logs messages to console and an optional file.
    No credentials or external APIs required. Used as default fallback.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.log_file = (config or {}).get("sms_log_file", "violations/sms_log.json")

    def send(self, phone_number, message, metadata=None):
        record = self._create_record(
            phone_number, message, "MOCK_DELIVERED", "DemoSMSGateway", metadata
        )
        print(f"[SMS-Demo] -> {phone_number}: {message[:80]}...")

        # Persist to JSON log file
        try:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
            existing = []
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(record)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SMS-Demo] Log write error: {e}")

        return record


class TwilioSMSGateway(BaseSMSGateway):
    """
    Twilio SMS gateway integration.
    Requires: account_sid, auth_token, from_number in config.
    """

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config or {}
        self.account_sid = cfg.get("account_sid", "")
        self.auth_token = cfg.get("auth_token", "")
        self.from_number = cfg.get("from_number", "")

    def send(self, phone_number, message, metadata=None):
        if not all([self.account_sid, self.auth_token, self.from_number]):
            print("[SMS-Twilio] Missing credentials. Falling back to MOCK_SENT.")
            return self._create_record(
                phone_number, message, "MOCK_SENT (no credentials)", "TwilioSMSGateway", metadata
            )

        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            data = urllib.parse.urlencode({
                "To": phone_number,
                "From": self.from_number,
                "Body": message[:1600]
            }).encode("utf-8")

            req = urllib.request.Request(url, data=data, method="POST")
            # Basic auth
            import base64
            credentials = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()
            req.add_header("Authorization", f"Basic {credentials}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                sid = resp_data.get("sid", "unknown")
                print(f"[SMS-Twilio] Sent to {phone_number}, SID: {sid}")
                return self._create_record(
                    phone_number, message, "SENT", "TwilioSMSGateway",
                    {**(metadata or {}), "twilio_sid": sid}
                )
        except Exception as e:
            print(f"[SMS-Twilio] Send failed: {e}")
            return self._create_record(
                phone_number, message, f"FAILED ({e})", "TwilioSMSGateway", metadata
            )


class Fast2SMSGateway(BaseSMSGateway):
    """
    Fast2SMS Indian bulk SMS gateway integration.
    Requires: api_key in config.
    DLT route for transactional messages (Indian TRAI compliance).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = (config or {}).get("api_key", "")
        self.sender_id = (config or {}).get("sender_id", "TRFSNT")
        self.route = (config or {}).get("route", "dlt")

    def send(self, phone_number, message, metadata=None):
        if not self.api_key:
            print("[SMS-Fast2SMS] Missing API key. Falling back to MOCK_SENT.")
            return self._create_record(
                phone_number, message, "MOCK_SENT (no API key)", "Fast2SMSGateway", metadata
            )

        try:
            # Strip country code for Fast2SMS (expects 10-digit Indian numbers)
            clean_number = phone_number.replace("+91", "").replace(" ", "").strip()
            if len(clean_number) > 10:
                clean_number = clean_number[-10:]

            url = "https://www.fast2sms.com/dev/bulkV2"
            params = urllib.parse.urlencode({
                "authorization": self.api_key,
                "route": self.route,
                "sender_id": self.sender_id,
                "message": message[:1000],
                "language": "english",
                "flash": 0,
                "numbers": clean_number,
            })

            req = urllib.request.Request(f"{url}?{params}")
            req.add_header("cache-control", "no-cache")

            with urllib.request.urlopen(req, timeout=10) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                if resp_data.get("return"):
                    print(f"[SMS-Fast2SMS] Sent to {clean_number}")
                    return self._create_record(
                        phone_number, message, "SENT", "Fast2SMSGateway",
                        {**(metadata or {}), "request_id": resp_data.get("request_id", "")}
                    )
                else:
                    raise Exception(resp_data.get("message", "Unknown error"))
        except Exception as e:
            print(f"[SMS-Fast2SMS] Send failed: {e}")
            return self._create_record(
                phone_number, message, f"FAILED ({e})", "Fast2SMSGateway", metadata
            )


def create_sms_gateway(config=None):
    """
    Factory function to create the appropriate SMS gateway based on config.

    Args:
        config (dict): SMS configuration with 'provider' key.

    Returns:
        BaseSMSGateway: Configured SMS gateway instance.
    """
    cfg = config or {}
    provider = cfg.get("provider", "demo").lower()

    if provider == "twilio":
        return TwilioSMSGateway(cfg)
    elif provider == "fast2sms":
        return Fast2SMSGateway(cfg)
    else:
        return DemoSMSGateway(cfg)
