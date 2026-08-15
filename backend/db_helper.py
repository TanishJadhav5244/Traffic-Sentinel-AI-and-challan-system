import os
import csv
import datetime
import uuid
import cv2
import pandas as pd

# Import RTO and Challan generator utilities
from backend.rto_helper import query_rto
from backend.challan_generator import generate_challan_ticket

class ViolationDatabase:
    def __init__(self, csv_log_path="violations/violations_log.csv", crop_dir="violations/crops"):
        self.csv_log_path = csv_log_path
        self.crop_dir = crop_dir
        self._initialize_storage()

    def _initialize_storage(self):
        """Creates output directories and CSV file if they don't exist."""
        os.makedirs(self.crop_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
        
        self.headers = [
            "violation_id", "timestamp", "plate_text", "ocr_confidence", "helmet_status", 
            "plate_crop_path", "rider_crop_path", "owner_name", "vehicle_model", "challan_amount", "challan_path", "night_mode"
        ]
        
        # If CSV log does not exist, create it with headers
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        else:
            # Check if columns match and migrate if needed
            try:
                with open(self.csv_log_path, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    existing_headers = next(reader, [])
                if existing_headers != self.headers:
                    print("[DB] Existing database headers don't match the new schema. Migrating database...")
                    # Read records
                    df = pd.read_csv(self.csv_log_path)
                    # Add missing columns
                    for col in self.headers:
                        if col not in df.columns:
                            if col == "challan_amount":
                                df[col] = 1000.0
                            elif col == "owner_name":
                                df[col] = "Unknown Owner"
                            elif col == "vehicle_model":
                                df[col] = "Unknown Model"
                            elif col == "night_mode":
                                df[col] = False
                            else:
                                df[col] = ""
                    # Reorder and write back
                    df = df[self.headers]
                    df.to_csv(self.csv_log_path, index=False, encoding='utf-8')
                    print("[DB] Database migration completed successfully.")
            except Exception as e:
                print(f"[DB] Error checking/migrating database headers: {e}")

    def log_violation(self, frame_timestamp, plate_crop, rider_crop, plate_text, ocr_conf, 
                      helmet_status="no-helmet", night_mode=False):
        """
        Logs a single violation, queries RTO info, generates E-Challan, and saves crop images.
        """
        violation_id = str(uuid.uuid4())[:8]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Query RTO vehicle details
        rto_info = query_rto(plate_text)
        owner_name = rto_info.get("owner_name", "Unknown Owner")
        vehicle_model = rto_info.get("vehicle_model", "Unknown Model")
        challan_amount = 1000.0
        
        plate_filename = f"plate_{violation_id}.png"
        rider_filename = f"rider_{violation_id}.png"
        
        plate_path = os.path.join(self.crop_dir, plate_filename)
        rider_path = os.path.join(self.crop_dir, rider_filename)
        
        # Save images
        if plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(plate_path, plate_crop)
        else:
            plate_path = ""
            
        if rider_crop is not None and rider_crop.size > 0:
            cv2.imwrite(rider_path, rider_crop)
        else:
            rider_path = ""
            
        # Generate visual E-Challan ticket with crops embedded
        challan_path = ""
        if plate_crop is not None and plate_crop.size > 0:
            try:
                challan_path = generate_challan_ticket(
                    violation_id=violation_id,
                    timestamp=timestamp_str,
                    plate_text=plate_text,
                    rto_info=rto_info,
                    plate_crop=plate_crop,
                    rider_crop=rider_crop
                )
            except Exception as e:
                print(f"[DB] Error generating visual E-Challan: {e}")
                
        # Append to CSV
        record = {
            "violation_id": violation_id,
            "timestamp": timestamp_str,
            "plate_text": plate_text,
            "ocr_confidence": round(ocr_conf, 2),
            "helmet_status": helmet_status,
            "plate_crop_path": plate_path,
            "rider_crop_path": rider_path,
            "owner_name": owner_name,
            "vehicle_model": vehicle_model,
            "challan_amount": challan_amount,
            "challan_path": challan_path,
            "night_mode": night_mode
        }
        
        with open(self.csv_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)
            
        return record

    def get_all_violations(self):
        """Reads and returns all violations as a pandas DataFrame."""
        if not os.path.exists(self.csv_log_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(self.csv_log_path)
        except Exception:
            return pd.DataFrame()

    def clear_database(self):
        """Clears all records and deletes stored crop images and challans."""
        if os.path.exists(self.crop_dir):
            for file in os.listdir(self.crop_dir):
                file_path = os.path.join(self.crop_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Clear challans
        challans_dir = os.path.join(os.path.dirname(self.crop_dir), "challans")
        if os.path.exists(challans_dir):
            for file in os.listdir(challans_dir):
                file_path = os.path.join(challans_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Reset CSV file
        with open(self.csv_log_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def get_analytics_data(self):
        """
        Returns a dict of pre-aggregated DataFrames for the Analytics Dashboard.
        All DataFrames are safe (empty-checked) so callers don't need to guard.
        """
        df = self.get_all_violations()
        result = {
            "total": 0,
            "total_fines": 0.0,
            "daily_series": pd.DataFrame(),
            "hourly_series": pd.DataFrame(),
            "state_counts": pd.DataFrame(),
            "manufacturer_counts": pd.DataFrame(),
            "confidence_dist": pd.DataFrame(),
            "top_plates": pd.DataFrame(),
        }

        if df.empty:
            return result

        # Parse timestamps
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["_ts"])

        result["total"] = len(df)
        result["total_fines"] = float(df.get("challan_amount", pd.Series([1000.0] * len(df))).sum())

        # --- Daily trend (last 30 days) ---
        df["_date"] = df["_ts"].dt.date
        daily = df.groupby("_date").size().reset_index(name="violations")
        daily.columns = ["Date", "Violations"]
        daily["Date"] = pd.to_datetime(daily["Date"])
        result["daily_series"] = daily

        # --- Hourly distribution (0-23) ---
        df["_hour"] = df["_ts"].dt.hour
        hourly = df.groupby("_hour").size().reset_index(name="violations")
        # Fill missing hours with 0
        all_hours = pd.DataFrame({"_hour": range(24)})
        hourly = all_hours.merge(hourly, on="_hour", how="left").fillna(0)
        hourly.columns = ["Hour", "Violations"]
        hourly["Violations"] = hourly["Violations"].astype(int)
        result["hourly_series"] = hourly

        # --- State breakdown (first 2 chars of plate) ---
        if "plate_text" in df.columns:
            states = (
                df["plate_text"]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
                .str[:2]
            )
            states = states[states.str.match(r"^[A-Z]{2}$", na=False)]
            if not states.empty:
                sc = states.value_counts().reset_index()
                sc.columns = ["State", "Violations"]
                result["state_counts"] = sc

        # --- Manufacturer breakdown ---
        if "vehicle_model" in df.columns:
            brands = (
                df["vehicle_model"]
                .dropna()
                .astype(str)
                .apply(lambda x: x.split()[0] if x.strip() else "Unknown")
            )
            bc = brands.value_counts().reset_index()
            bc.columns = ["Manufacturer", "Violations"]
            result["manufacturer_counts"] = bc

        # --- OCR confidence distribution ---
        if "ocr_confidence" in df.columns:
            conf = df["ocr_confidence"].dropna()
            if not conf.empty:
                bins = pd.cut(conf, bins=[0, 0.3, 0.5, 0.7, 0.85, 1.01],
                              labels=["<30%", "30-50%", "50-70%", "70-85%", ">85%"],
                              right=False)
                cd = bins.value_counts().sort_index().reset_index()
                cd.columns = ["Confidence Band", "Count"]
                result["confidence_dist"] = cd

        # --- Top repeat plates ---
        if "plate_text" in df.columns:
            tp = (
                df["plate_text"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )
            tp.columns = ["Plate", "Occurrences"]
            result["top_plates"] = tp

        return result
