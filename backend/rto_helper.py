import hashlib
import re

# Database of predefined mock vehicle records
MOCK_RTO_REGISTRY = {
    "MH12DE5678": {
        "owner_name": "Rajesh Kumar",
        "vehicle_make": "TVS",
        "vehicle_model": "TVS Jupiter 125",
        "fuel_type": "Petrol",
        "insurance_status": "Active (Insured)",
        "registration_date": "2022-04-12",
        "status": "Active"
    },
    "MH12AB1234": {
        "owner_name": "Ananya Sharma",
        "vehicle_make": "Honda",
        "vehicle_model": "Honda Activa 6G",
        "fuel_type": "Petrol",
        "insurance_status": "Active (Insured)",
        "registration_date": "2021-10-18",
        "status": "Active"
    },
    "DL3CAY1111": {
        "owner_name": "Amit Verma",
        "vehicle_make": "Royal Enfield",
        "vehicle_model": "Classic 350",
        "fuel_type": "Petrol",
        "insurance_status": "Active (Insured)",
        "registration_date": "2020-11-05",
        "status": "Active"
    },
    "KA03MG9999": {
        "owner_name": "Vikram Rao",
        "vehicle_make": "Ather Energy",
        "vehicle_model": "Ather 450X Gen 3",
        "fuel_type": "Electric",
        "insurance_status": "Active (Insured)",
        "registration_date": "2023-01-20",
        "status": "Active"
    },
    "HR26BP0007": {
        "owner_name": "Harpreet Singh",
        "vehicle_make": "Suzuki",
        "vehicle_model": "Suzuki Access 125",
        "fuel_type": "Petrol",
        "insurance_status": "Expired",
        "registration_date": "2019-08-25",
        "status": "Active"
    }
}

# Lists of words for generating mock records deterministically
FIRST_NAMES = ["Sanjay", "Priya", "Rahul", "Deepak", "Sneha", "Rohan", "Neha", "Abhishek", "Karan", "Pooja", "Arjun", "Aditi"]
LAST_NAMES = ["Patel", "Sharma", "Joshi", "Gupta", "Singh", "Mehta", "Nair", "Reddy", "Choudhury", "Verma", "Rao", "Mishra"]
VEHICLES = [
    ("Honda", "Activa 6G", "Petrol"),
    ("Honda", "Dio", "Petrol"),
    ("TVS", "Jupiter", "Petrol"),
    ("TVS", "Ntorq 125", "Petrol"),
    ("TVS", "iQube", "Electric"),
    ("Suzuki", "Access 125", "Petrol"),
    ("Hero", "Splendor Plus", "Petrol"),
    ("Hero", "HF Deluxe", "Petrol"),
    ("Yamaha", "Fascino 125", "Petrol"),
    ("Bajaj", "Chetak", "Electric"),
    ("Ather Energy", "Ather 450X", "Electric"),
    ("Ola Electric", "Ola S1 Pro", "Electric"),
    ("Royal Enfield", "Classic 350", "Petrol")
]

def query_rto(plate_number):
    """
    Queries the RTO vehicle registry database.
    If the plate is not predefined, details are generated deterministically based on the plate hash.
    
    Args:
        plate_number (str): License plate text.
        
    Returns:
        dict: Vehicle registry details.
    """
    if not plate_number:
        return {
            "owner_name": "Unknown Owner",
            "vehicle_make": "Unknown Make",
            "vehicle_model": "Unknown Model",
            "fuel_type": "Unknown",
            "insurance_status": "Unknown",
            "registration_date": "N/A",
            "status": "Unknown"
        }
        
    # Clean the plate string: uppercase, remove symbols/spaces
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
    
    # Check if predefined
    if clean_plate in MOCK_RTO_REGISTRY:
        return MOCK_RTO_REGISTRY[clean_plate]
        
    # Generate deterministically using plate MD5 hash
    plate_hash = hashlib.md5(clean_plate.encode('utf-8')).hexdigest()
    hash_int = int(plate_hash, 16)
    
    # Pick owner
    fn_idx = (hash_int) % len(FIRST_NAMES)
    ln_idx = (hash_int >> 4) % len(LAST_NAMES)
    owner = f"{FIRST_NAMES[fn_idx]} {LAST_NAMES[ln_idx]}"
    
    # Pick vehicle
    v_idx = (hash_int >> 8) % len(VEHICLES)
    make, model, fuel = VEHICLES[v_idx]
    
    # Pick insurance status (80% Active, 20% Expired)
    ins_status = "Active (Insured)" if ((hash_int >> 12) % 10) < 8 else "Expired"
    
    # Pick registration year (2018-2025) and month/day
    reg_year = 2018 + ((hash_int >> 16) % 8)
    reg_month = 1 + ((hash_int >> 20) % 12)
    reg_day = 1 + ((hash_int >> 24) % 28)
    reg_date = f"{reg_year:04d}-{reg_month:02d}-{reg_day:02d}"
    
    return {
        "owner_name": owner,
        "vehicle_make": make,
        "vehicle_model": model,
        "fuel_type": fuel,
        "insurance_status": ins_status,
        "registration_date": reg_date,
        "status": "Active"
    }
