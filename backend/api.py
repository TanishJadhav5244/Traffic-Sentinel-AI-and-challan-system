import os
import cv2
import yaml
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from backend.detector import VehicleHelmetDetector
from backend.ocr_engine import LicensePlateOCR
from backend.image_enhancer import enhance
from backend.rto_helper import RTORegistry
from backend.db_helper import ViolationDatabase
from backend.challan_generator import EChallanGenerator
from backend.notifier import ChallanNotifier
from backend.tracker import VehicleTracker

# Load YAML configuration
config_path = "config.yaml"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

app = FastAPI(
    title="Traffic Sentinel AI — REST API",
    description="Enterprise REST API for Automated Helmet Violation Detection, License Plate OCR, RTO Lookup & E-Challan Management",
    version="2.0.0"
)

# Initialize Core Services
detector = VehicleHelmetDetector(config)
ocr = LicensePlateOCR(config)
rto = RTORegistry()
db = ViolationDatabase(config.get("database", {}).get("db_path", "violations.db"))
challan_gen = EChallanGenerator(config)
notifier = ChallanNotifier(config)
tracker = VehicleTracker(
    cooldown_seconds=config.get("video", {}).get("cooldown_seconds", 300)
)

class StatusUpdatePayload(BaseModel):
    status: str

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "system": "Traffic Sentinel AI",
        "version": "2.0.0",
        "status": "ONLINE",
        "model_loaded": detector.model is not None,
        "ocr_engine": ocr.engine_type
    }

@app.post("/api/v1/scan/image")
async def scan_image(
    file: UploadFile = File(...),
    enhance_low_light: bool = Query(True, description="Enable Retinex/CLAHE night vision enhancement"),
    auto_generate_challan: bool = Query(True, description="Auto-generate E-Challan ticket on violation")
):
    """
    Ingests a single traffic camera image frame, detects helmet violations,
    performs license plate OCR, fetches RTO records, and creates violation tickets.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    # 1. Detection
    detections = detector.detect(img)
    violations = detector.associate_violations(detections)

    results = []

    for v in violations:
        plate_crop = v["plate_crop"] if "plate_crop" in v else None
        
        # Crop plate bounding box if crop not directly provided
        if plate_crop is None:
            p_box = v["plate"]["box"]
            h, w, _ = img.shape
            x1, y1, x2, y2 = max(0, p_box[0]), max(0, p_box[1]), min(w, p_box[2]), min(h, p_box[3])
            plate_crop = img[y1:y2, x1:x2]

        if plate_crop is None or plate_crop.size == 0:
            continue

        # 2. Low light enhancement if enabled
        enhanced_crop = plate_crop
        was_enhanced = False
        if enhance_low_light:
            enh_res = enhance(plate_crop)
            enhanced_crop = enh_res.get("enhanced", plate_crop)
            was_enhanced = enh_res.get("was_low_light", False)

        # 3. License plate OCR
        ocr_result = ocr.read_plate(enhanced_crop)
        plate_text = ocr_result.get("text", "UNKNOWN")

        # Check duplicate suppression
        if tracker.is_duplicate_plate(plate_text):
            continue

        # 4. RTO lookup
        rto_info = rto.lookup(plate_text)

        # 5. Crop saving & E-Challan ticket generation
        head_box = v["no_helmet"]["box"]
        hx1, hy1, hx2, hy2 = max(0, head_box[0]), max(0, head_box[1]), min(img.shape[1], head_box[2]), min(img.shape[0], head_box[3])
        head_crop = img[hy1:hy2, hx1:hx2]

        crop_paths = db.save_violation_crops(
            head_crop=head_crop if head_crop.size > 0 else plate_crop,
            plate_crop=enhanced_crop,
            full_frame=img,
            plate_text=plate_text
        )

        challan_path = None
        challan_id = None

        if auto_generate_challan:
            challan_path, challan_id = challan_gen.generate(
                plate_number=plate_text,
                rto_details=rto_info,
                head_crop=head_crop if head_crop.size > 0 else plate_crop,
                plate_crop=enhanced_crop,
                location=config.get("system", {}).get("location", "Intersection Cam #04, MG Road")
            )
            tracker.mark_plate_ticketed(plate_text)

            # Record in DB
            db.insert_violation(
                plate_number=plate_text,
                confidence=float(v["no_helmet"].get("conf", 0.0)),
                owner_name=rto_info.get("owner_name", "Unknown"),
                location=config.get("system", {}).get("location", "Intersection Cam #04, MG Road"),
                challan_id=challan_id,
                challan_path=challan_path,
                head_crop_path=crop_paths.get("head_crop"),
                plate_crop_path=crop_paths.get("plate_crop"),
                full_frame_path=crop_paths.get("full_frame")
            )

            # Send Notification log
            notifier.send_email_challan(
                recipient_email=f"{plate_text.lower()}@owner-contact.in",
                owner_name=rto_info.get("owner_name", "Vehicle Owner"),
                plate_number=plate_text,
                challan_id=challan_id,
                fine_amount=config.get("challan", {}).get("fine_amount", 1000),
                challan_image_path=challan_path
            )

        results.append({
            "plate_number": plate_text,
            "ocr_confidence": ocr_result.get("confidence", 0.0),
            "is_valid_format": ocr_result.get("is_valid", False),
            "enhanced_low_light": was_enhanced,
            "rto_info": rto_info,
            "challan_id": challan_id,
            "challan_path": challan_path
        })

    return {
        "status": "SUCCESS",
        "detections_count": len(violations),
        "violations": results
    }

@app.get("/api/v1/violations")
def get_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None
):
    """Fetches paginated violation records with optional plate search and status filter."""
    records, total_count = db.get_violations_paginated(
        page=page,
        page_size=page_size,
        search_query=search,
        status_filter=status
    )
    return {
        "page": page,
        "page_size": page_size,
        "total_records": total_count,
        "violations": records
    }

@app.get("/api/v1/violations/{violation_id}")
def get_violation_detail(violation_id: int):
    """Retrieves single violation record details."""
    rec = db.get_violation_by_id(violation_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Violation record not found.")
    return rec

@app.patch("/api/v1/violations/{violation_id}/status")
def update_violation_status(violation_id: int, payload: StatusUpdatePayload):
    """Updates violation ticket status (e.g. PENDING, PAID, CANCELLED)."""
    valid_statuses = ["PENDING", "PAID", "CANCELLED"]
    if payload.status.upper() not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of {valid_statuses}")

    updated = db.update_status(violation_id, payload.status.upper())
    if not updated:
        raise HTTPException(status_code=404, detail="Violation record not found.")
    return {"status": "SUCCESS", "violation_id": violation_id, "new_status": payload.status.upper()}

@app.get("/api/v1/analytics/stats")
def get_analytics_summary():
    """Returns analytics aggregations and violation count statistics."""
    summary = db.get_summary_stats()
    hourly = db.get_hourly_analytics()
    manufacturers = db.get_manufacturer_stats()
    return {
        "summary": summary,
        "hourly_distribution": hourly,
        "manufacturer_distribution": manufacturers
    }

@app.get("/api/v1/challan/file/{filename}")
def get_challan_image(filename: str):
    """Serves generated E-Challan PNG image file."""
    path = os.path.join("violations", "challans", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Challan image file not found.")
    return FileResponse(path, media_type="image/png")

@app.post("/api/v1/challan/{challan_id}/pay")
def pay_challan(challan_id: str, method: str = Query("UPI_ONLINE", description="Payment method")):
    """Processes digital payment for an E-Challan ticket and generates confirmation receipt."""
    from backend.challan_payment import EChallanPaymentGateway
    gateway = EChallanPaymentGateway(db)
    res = gateway.process_payment(challan_id, payment_method=method)
    if res.get("status") == "ERROR":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res
