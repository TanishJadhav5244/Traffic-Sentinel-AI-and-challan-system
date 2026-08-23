import re
import urllib.request
import urllib.error
import json

# Predefined demo registry (only used if explicit demo mode is enabled)
DEMO_RTO_REGISTRY = {
    "MH12DE5678": {
        "owner_name": "[DEMO] Rajesh Kumar",
        "vehicle_make": "TVS",
        "vehicle_model": "TVS Jupiter 125",
        "fuel_type": "Petrol",
        "insurance_status": "Active (Insured)",
        "registration_date": "2022-04-12",
        "status": "Active [DEMO]",
        "api_source": "Demo Fallback Registry"
    },
    "MH12AB1234": {
        "owner_name": "[DEMO] Ananya Sharma",
        "vehicle_make": "Honda",
        "vehicle_model": "Honda Activa 6G",
        "fuel_type": "Petrol",
        "insurance_status": "Active (Insured)",
        "registration_date": "2021-10-18",
        "status": "Active [DEMO]",
        "api_source": "Demo Fallback Registry"
    },
    "DL3CAY1111": {
        "owner_name": "[DEMO] Amit Verma",
        "vehicle_make": "Royal Enfield",
        "vehicle_model": "Classic 350",
        "fuel_type": "Petrol",
        "insurance_status": "Active (Insured)",
        "registration_date": "2020-11-05",
        "status": "Active [DEMO]",
        "api_source": "Demo Fallback Registry"
    }
}


def query_rto(plate_number, config=None):
    """
    Queries the RTO vehicle registry database or configured RTO API.
    
    Returns real API response with status reporting, or clearly labels
    unavailable data when no API key/endpoint is configured.
    Never fabricates random/fake vehicle details.
    
    Args:
        plate_number (str): License plate text.
        config (dict, optional): Configuration dictionary loaded from config.yaml.
        
    Returns:
        dict: Vehicle registry details and lookup status metadata.
    """
    rto_config = (config or {}).get("rto", {})
    
    if not plate_number or not plate_number.strip():
        return {
            "owner_name": "[Invalid Input]",
            "vehicle_make": "[N/A]",
            "vehicle_model": "[N/A]",
            "fuel_type": "[N/A]",
            "insurance_status": "[N/A]",
            "registration_date": "[N/A]",
            "status": "Invalid Input",
            "lookup_status": "error_invalid_input"
        }
        
    # Clean the plate string: uppercase, remove symbols/spaces
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
    
    api_url = rto_config.get("api_url", "").strip()
    api_key = rto_config.get("api_key", "").strip()
    timeout = float(rto_config.get("timeout_seconds", 5.0))
    demo_fallback = rto_config.get("demo_fallback", False)
    
    # 1. Real API HTTP Lookup Flow
    if api_url:
        return _fetch_from_api(clean_plate, api_url, api_key, timeout)

    # 2. Demo Fallback Mode (Explicit Opt-in only)
    if demo_fallback and clean_plate in DEMO_RTO_REGISTRY:
        return DEMO_RTO_REGISTRY[clean_plate]

    # 3. Clean Boundary: Unconfigured / Missing API Key (Never fabricate data)
    return {
        "owner_name": "[No RTO API Configured]",
        "vehicle_make": "[Data Unavailable]",
        "vehicle_model": "[Data Unavailable]",
        "fuel_type": "[Data Unavailable]",
        "insurance_status": "[Data Unavailable]",
        "registration_date": "[N/A]",
        "status": "Unconfigured",
        "lookup_status": "api_not_configured",
        "message": "Real vehicle lookup requires an active RTO API endpoint & key in config.yaml."
    }


def _fetch_from_api(plate_number, api_url, api_key, timeout=5.0):
    """Makes a real HTTP request to the configured RTO API endpoint."""
    url = f"{api_url}?plate={plate_number}" if "?" not in api_url else f"{api_url}&plate={plate_number}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "TrafficSentinelAI/1.0")
    req.add_header("Accept", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("X-API-Key", api_key)
        
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            body = response.read().decode('utf-8')
            data = json.loads(body)
            
            return {
                "owner_name": data.get("owner_name") or data.get("owner") or "[Unknown]",
                "vehicle_make": data.get("vehicle_make") or data.get("make") or "[Unknown]",
                "vehicle_model": data.get("vehicle_model") or data.get("model") or "[Unknown]",
                "fuel_type": data.get("fuel_type") or data.get("fuel") or "[Unknown]",
                "insurance_status": data.get("insurance_status") or data.get("insurance") or "[Unknown]",
                "registration_date": data.get("registration_date") or data.get("reg_date") or "[N/A]",
                "status": data.get("status", "Active"),
                "lookup_status": "success",
                "api_source": api_url
            }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {
                "owner_name": "[Registration Not Found]",
                "vehicle_make": "[N/A]",
                "vehicle_model": "[N/A]",
                "fuel_type": "[N/A]",
                "insurance_status": "[N/A]",
                "registration_date": "[N/A]",
                "status": "Nonexistent Plate",
                "lookup_status": "error_not_found"
            }
        elif e.code == 429:
            return {
                "owner_name": "[API Rate Limit Exceeded]",
                "vehicle_make": "[Data Unavailable]",
                "vehicle_model": "[Data Unavailable]",
                "fuel_type": "[Data Unavailable]",
                "insurance_status": "[Data Unavailable]",
                "registration_date": "[N/A]",
                "status": "Rate Limited",
                "lookup_status": "error_rate_limit"
            }
        elif e.code in (401, 403):
            return {
                "owner_name": "[API Unauthorized / Key Invalid]",
                "vehicle_make": "[Data Unavailable]",
                "vehicle_model": "[Data Unavailable]",
                "fuel_type": "[Data Unavailable]",
                "insurance_status": "[Data Unavailable]",
                "registration_date": "[N/A]",
                "status": "Unauthorized",
                "lookup_status": "error_unauthorized"
            }
        else:
            return {
                "owner_name": f"[API Error {e.code}]",
                "vehicle_make": "[Data Unavailable]",
                "vehicle_model": "[Data Unavailable]",
                "fuel_type": "[Data Unavailable]",
                "insurance_status": "[Data Unavailable]",
                "registration_date": "[N/A]",
                "status": f"HTTP {e.code}",
                "lookup_status": "error_http"
            }

    except urllib.error.URLError as e:
        return {
            "owner_name": "[Connection Failed]",
            "vehicle_make": "[Data Unavailable]",
            "vehicle_model": "[Data Unavailable]",
            "fuel_type": "[Data Unavailable]",
            "insurance_status": "[Data Unavailable]",
            "registration_date": "[N/A]",
            "status": "Connection Error",
            "lookup_status": "error_connection",
            "message": str(e.reason)
        }

    except TimeoutError:
        return {
            "owner_name": "[API Request Timed Out]",
            "vehicle_make": "[Data Unavailable]",
            "vehicle_model": "[Data Unavailable]",
            "fuel_type": "[Data Unavailable]",
            "insurance_status": "[Data Unavailable]",
            "registration_date": "[N/A]",
            "status": "Timeout",
            "lookup_status": "error_timeout"
        }

    except Exception as e:
        return {
            "owner_name": "[Lookup Failed]",
            "vehicle_make": "[Data Unavailable]",
            "vehicle_model": "[Data Unavailable]",
            "fuel_type": "[Data Unavailable]",
            "insurance_status": "[Data Unavailable]",
            "registration_date": "[N/A]",
            "status": "Error",
            "lookup_status": "error_general",
            "message": str(e)
        }


class RTORegistry:
    """Wrapper class around query_rto for object-oriented callers."""
    def __init__(self, config=None):
        self.config = config

    def lookup(self, plate_number):
        return query_rto(plate_number, self.config)
