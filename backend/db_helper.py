import os
import re
import csv
import datetime
import uuid
import cv2
import pandas as pd

# Import RTO and Challan generator utilities
from backend.rto_helper import query_rto
from backend.challan_generator import generate_challan_ticket

class ViolationDatabase:
    def __init__(self, csv_log_path="violations/violations_log.csv", crop_dir="violations/crops", config=None):
        self.csv_log_path = csv_log_path
        self.crop_dir = crop_dir
        self.config = config or {}
        self._initialize_storage()

    def _initialize_storage(self):
        """Creates output directories and CSV file if they don't exist."""
        os.makedirs(self.crop_dir, exist_ok=True)
        csv_dir = os.path.dirname(self.csv_log_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        
        self.headers = [
            "violation_id", "timestamp", "plate_text", "ocr_confidence", "helmet_status", 
            "violation_type", "speed_recorded", "camera_id", "location",
            "plate_crop_path", "rider_crop_path", "owner_name", "vehicle_model", 
            "challan_amount", "challan_path", "night_mode", "status"
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
                            elif col == "violation_type":
                                df[col] = "No Helmet"
                            elif col == "speed_recorded":
                                df[col] = 0.0
                            elif col == "camera_id":
                                df[col] = "CAM-01"
                            elif col == "location":
                                df[col] = "MG Road Crossing"
                            elif col == "night_mode":
                                df[col] = False
                            elif col == "status":
                                df[col] = "Pending"
                            else:
                                df[col] = ""
                    # Reorder and write back
                    df = df[self.headers]
                    df.to_csv(self.csv_log_path, index=False, encoding='utf-8')
                    print("[DB] Database migration completed successfully.")
            except Exception as e:
                print(f"[DB] Error checking/migrating database headers: {e}")

    def log_violation(self, frame_timestamp, plate_crop, rider_crop, plate_text, ocr_conf, 
                      helmet_status="no-helmet", night_mode=False,
                      violation_type="No Helmet", speed_recorded=0.0,
                      camera_id="CAM-01", location="MG Road Crossing",
                      challan_amount=None):
        """
        Logs a single violation, queries RTO info, generates E-Challan, and saves crop images.
        """
        violation_id = str(uuid.uuid4())[:8]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Query RTO vehicle details
        rto_info = query_rto(plate_text, config=self.config)
        owner_name = rto_info.get("owner_name", "Unknown Owner")
        vehicle_model = rto_info.get("vehicle_model", "Unknown Model")

        # Determine fine amount according to violation types if not explicitly set
        if challan_amount is None or challan_amount <= 0:
            fine = 0.0
            vt_lower = violation_type.lower()
            if "helmet" in vt_lower:
                fine += 1000.0
            if "triple" in vt_lower:
                fine += 1000.0
            if "speed" in vt_lower:
                fine += 2000.0
            challan_amount = max(fine, 1000.0)
        
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
            "violation_type": violation_type,
            "speed_recorded": round(float(speed_recorded), 1),
            "camera_id": camera_id,
            "location": location,
            "plate_crop_path": plate_path,
            "rider_crop_path": rider_path,
            "owner_name": owner_name,
            "vehicle_model": vehicle_model,
            "challan_amount": float(challan_amount),
            "challan_path": challan_path,
            "night_mode": night_mode,
            "status": "Pending"
        }
        
        with open(self.csv_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(record)
            
        return record

    def get_all_violations(self):
        """Reads and returns all violations as a pandas DataFrame."""
        if not os.path.exists(self.csv_log_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.csv_log_path)
            if "status" not in df.columns:
                df["status"] = "Pending"
            return df
        except Exception:
            return pd.DataFrame()

    def update_violation_status(self, violation_id: str, new_status: str) -> bool:
        """Updates the status of a specific violation record (e.g., Pending, Paid, Disputed)."""
        if not os.path.exists(self.csv_log_path):
            return False
        try:
            df = pd.read_csv(self.csv_log_path)
            if "violation_id" not in df.columns or "status" not in df.columns:
                return False
            mask = df["violation_id"].astype(str) == str(violation_id)
            if mask.any():
                df.loc[mask, "status"] = new_status
                df.to_csv(self.csv_log_path, index=False, encoding='utf-8')
                return True
            return False
        except Exception as e:
            print(f"[DB] Error updating violation status: {e}")
            return False

    def update_status(self, violation_id, new_status: str) -> bool:
        """Alias for update_violation_status, used by api.py."""
        return self.update_violation_status(str(violation_id), new_status)

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
            "violation_types": pd.DataFrame(),
            "speed_dist": pd.DataFrame(),
            "weekday_hour_matrix": pd.DataFrame(),
            "camera_hotspots": pd.DataFrame(),
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

        # --- Weekday × Hourly Matrix Heatmap ---
        df["_day_name"] = df["_ts"].dt.day_name()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        matrix_data = []
        for day in days_order:
            day_df = df[df["_day_name"] == day]
            counts = [int((day_df["_hour"] == h).sum()) for h in range(24)]
            matrix_data.append(counts)
        result["weekday_hour_matrix"] = pd.DataFrame(matrix_data, index=days_order, columns=[f"{h:02d}:00" for h in range(24)])

        # --- Violation Type Breakdown ---
        if "violation_type" in df.columns:
            vt_series = df["violation_type"].fillna("No Helmet").astype(str)
            # Normalize composite labels for aggregation
            parsed_types = []
            for item in vt_series:
                if "triple" in item.lower() and "helmet" in item.lower():
                    parsed_types.append("No Helmet + Triple Riding")
                elif "triple" in item.lower():
                    parsed_types.append("Triple Riding")
                elif "speed" in item.lower():
                    parsed_types.append("Over-Speeding")
                else:
                    parsed_types.append("No Helmet")
            vt_df = pd.Series(parsed_types).value_counts().reset_index()
            vt_df.columns = ["Violation Type", "Count"]
            result["violation_types"] = vt_df
        else:
            result["violation_types"] = pd.DataFrame([{"Violation Type": "No Helmet", "Count": len(df)}])

        # --- Vehicle Speed Distribution ---
        if "speed_recorded" in df.columns:
            speeds = pd.to_numeric(df["speed_recorded"], errors="coerce").dropna()
            # Non-zero speeds
            active_speeds = speeds[speeds > 0]
            if not active_speeds.empty:
                speed_bins = pd.cut(active_speeds, bins=[0, 40, 60, 80, 100, 150],
                                    labels=["<40 km/h", "40-60 km/h (City)", "60-80 km/h (High)", "80-100 km/h (Speeding)", ">100 km/h (Extreme)"],
                                    right=False)
                spd_counts = speed_bins.value_counts().sort_index().reset_index()
                spd_counts.columns = ["Speed Range", "Vehicle Count"]
                result["speed_dist"] = spd_counts

        # --- Camera Location Hotspots ---
        if "location" in df.columns:
            locs = df["location"].fillna("MG Road Crossing").astype(str).value_counts().reset_index()
            locs.columns = ["Location", "Violations"]
            result["camera_hotspots"] = locs

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

    def get_violations_paginated(self, page=1, page_size=20, search_query=None, status_filter=None):
        """Returns paginated list of violation records with optional filtering."""
        df = self.get_all_violations()
        if df.empty:
            return [], 0

        if search_query:
            sq = str(search_query).strip().lower()
            df = df[df["plate_text"].astype(str).str.lower().str.contains(sq, na=False)]

        if status_filter:
            sf = str(status_filter).strip().upper()
            df = df[df["status"].astype(str).str.upper() == sf]

        total_count = len(df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        records = df.iloc[start_idx:end_idx].to_dict(orient="records")
        return records, total_count

    def get_violation_by_id(self, violation_id):
        """Finds a single violation record by violation_id."""
        df = self.get_all_violations()
        if df.empty:
            return None
        match = df[df["violation_id"].astype(str) == str(violation_id)]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def get_summary_stats(self):
        """Returns aggregate metrics summary."""
        df = self.get_all_violations()
        if df.empty:
            return {"total_violations": 0, "total_fines_inr": 0, "pending": 0, "paid": 0}
        
        total = len(df)
        fines = float(df.get("challan_amount", pd.Series([1000]*total)).sum())
        statuses = df.get("status", pd.Series(["PENDING"]*total)).astype(str).str.upper()
        pending = int((statuses == "PENDING").sum())
        paid = int((statuses == "PAID").sum())

        return {
            "total_violations": total,
            "total_fines_inr": fines,
            "pending": pending,
            "paid": paid
        }

    def get_hourly_analytics(self):
        """Returns hourly distribution dict."""
        analytics = self.get_analytics_data()
        df_hourly = analytics.get("hourly_series", pd.DataFrame())
        if df_hourly.empty:
            return []
        return df_hourly.to_dict(orient="records")

    def get_manufacturer_stats(self):
        """Returns manufacturer counts dict."""
        analytics = self.get_analytics_data()
        df_mfg = analytics.get("manufacturer_counts", pd.DataFrame())
        if df_mfg.empty:
            return []
        return df_mfg.to_dict(orient="records")

    def save_violation_crops(self, head_crop, plate_crop, full_frame, plate_text):
        """Saves head crop, plate crop, and full frame image to storage directory."""
        os.makedirs(self.crop_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        clean_text = re.sub(r'[^A-Z0-9]', '', str(plate_text).upper()) or "PLATE"

        head_path = os.path.join(self.crop_dir, f"rider_{clean_text}_{stamp}.jpg")
        plate_path = os.path.join(self.crop_dir, f"plate_{clean_text}_{stamp}.jpg")
        frame_path = os.path.join(self.crop_dir, f"frame_{clean_text}_{stamp}.jpg")

        if head_crop is not None and head_crop.size > 0:
            cv2.imwrite(head_path, head_crop)
        if plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(plate_path, plate_crop)
        if full_frame is not None and full_frame.size > 0:
            cv2.imwrite(frame_path, full_frame)

        return {
            "head_crop": head_path,
            "plate_crop": plate_path,
            "full_frame": frame_path
        }

    def add_violation(self, plate_text, ocr_confidence, helmet_status="No-Helmet",
                      plate_crop_path="", rider_crop_path="", owner_name="Unknown",
                      vehicle_model="Motorcycle", challan_amount=1000.0,
                      challan_path="", night_mode=False, violation_type="No Helmet",
                      speed_recorded=0.0, camera_id="CAM-01", location="MG Road Crossing"):
        """Adds a violation record directly from pre-computed fields (used by API)."""
        violation_id = str(uuid.uuid4())[:8]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "violation_id": violation_id,
            "timestamp": timestamp_str,
            "plate_text": plate_text,
            "ocr_confidence": round(float(ocr_confidence), 2),
            "helmet_status": helmet_status,
            "violation_type": violation_type,
            "speed_recorded": round(float(speed_recorded), 1),
            "camera_id": camera_id,
            "location": location,
            "plate_crop_path": plate_crop_path,
            "rider_crop_path": rider_crop_path,
            "owner_name": owner_name,
            "vehicle_model": vehicle_model,
            "challan_amount": float(challan_amount),
            "challan_path": challan_path,
            "night_mode": night_mode,
            "status": "Pending"
        }
        with open(self.csv_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(record)
        return record

    def insert_violation(self, plate_number, confidence, owner_name, location, challan_id, challan_path, head_crop_path, plate_crop_path, full_frame_path):
        """Wrapper method around add_violation to match API payload parameters."""
        return self.add_violation(
            plate_text=plate_number,
            ocr_confidence=confidence,
            helmet_status="No-Helmet",
            plate_crop_path=plate_crop_path or "",
            rider_crop_path=head_crop_path or "",
            owner_name=owner_name or "Unknown",
            vehicle_model="Motorcycle",
            challan_amount=1000.0,
            challan_path=challan_path or "",
            night_mode=False,
            violation_type="No Helmet",
            location=location or "MG Road Crossing"
        )

