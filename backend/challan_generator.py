import os
import cv2
import datetime
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_font(font_name="arial", size=14, bold=False):
    """Utility to load system fonts on Windows or fall back to default."""
    font_paths = []
    if bold:
        font_paths = [
            f"C:\\Windows\\Fonts\\{font_name}bd.ttf",
            "C:\\Windows\\Fonts\\consolab.ttf",
            "C:\\Windows\\Fonts\\tahomabd.ttf"
        ]
    else:
        font_paths = [
            f"C:\\Windows\\Fonts\\{font_name}.ttf",
            "C:\\Windows\\Fonts\\consola.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf"
        ]
        
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def draw_barcode(draw, x, y, width=150, height=40):
    """Draws a realistic mock barcode using PIL."""
    # Seed a simple pseudo-random pattern based on x coordinate
    pattern = [2, 1, 3, 1, 2, 4, 1, 2, 3, 2, 1, 1, 4, 2, 1, 3, 2]
    curr_x = x
    max_x = x + width
    idx = 0
    while curr_x < max_x:
        w = pattern[idx % len(pattern)] * 2
        # Draw black bar
        if idx % 2 == 0:
            draw.rectangle([curr_x, y, min(curr_x + w, max_x), y + height], fill=(0, 0, 0))
        curr_x += w + 1
        idx += 1

def generate_challan_ticket(violation_id, timestamp, plate_text, rto_info, plate_crop, rider_crop, output_dir="violations/challans"):
    """
    Generates a professional visual E-Challan PNG image and saves it to disk.
    
    Args:
        violation_id (str): Unique violation ID.
        timestamp (str): Timestamp of the violation.
        plate_text (str): Extracted plate text.
        rto_info (dict): RTO vehicle registry info.
        plate_crop (numpy.ndarray): OpenCV crop of the plate.
        rider_crop (numpy.ndarray): OpenCV crop of the rider.
        output_dir (str): Directory where the challan image will be saved.
        
    Returns:
        str: Absolute or relative path to the saved challan PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize blank white canvas (800 x 720)
    canvas_w, canvas_h = 800, 720
    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw gray outer border
    draw.rectangle([5, 5, canvas_w - 6, canvas_h - 6], outline=(150, 150, 150), width=3)
    
    # 2. Draw Top Official Header
    header_h = 100
    draw.rectangle([5, 5, canvas_w - 6, header_h], fill=(15, 23, 42)) # Dark slate gray
    
    font_title = get_font("arial", 22, bold=True)
    font_subtitle = get_font("arial", 12, bold=False)
    
    draw.text((30, 20), "DEPARTMENT OF TRAFFIC POLICE", fill=(255, 255, 255), font=font_title)
    draw.text((30, 52), "E-CHALLAN / OFFICIAL VIOLATION NOTICE", fill=(56, 189, 248), font=font_title) # light blue secondary
    draw.text((30, 80), "Government of India | Motor Vehicles Act, Section 129", fill=(148, 163, 184), font=font_subtitle)
    
    # 3. Draw Red Alert Banner
    banner_y = header_h + 10
    banner_h = 40
    draw.rectangle([15, banner_y, canvas_w - 16, banner_y + banner_h], fill=(239, 68, 68)) # red
    
    font_banner = get_font("arial", 16, bold=True)
    draw.text((30, banner_y + 10), "VIOLATION IDENTIFIED: RIDING WITHOUT SAFETY HELMET", fill=(255, 255, 255), font=font_banner)
    
    # 4. Details Section (Two columns grid)
    col_y = banner_y + banner_h + 20
    col_w = 360
    
    # Left Column: Ticket Details
    draw.rectangle([15, col_y, 15 + col_w, col_y + 220], outline=(226, 232, 240), width=1)
    # Left Header
    draw.rectangle([15, col_y, 15 + col_w, col_y + 30], fill=(241, 245, 249))
    font_section = get_font("arial", 12, bold=True)
    font_content_bold = get_font("arial", 11, bold=True)
    font_content = get_font("arial", 11, bold=False)
    
    draw.text((25, col_y + 8), "TICKET INFORMATION", fill=(15, 23, 42), font=font_section)
    
    ticket_details = [
        ("Challan Number", f"CH-{violation_id.upper()}"),
        ("Date & Time", timestamp),
        ("Location", "National Highway Camera 08 (Sector 62)"),
        ("Fine Amount", "INR 1,000.00"),
        ("Payment Status", "PENDING")
    ]
    
    y_offset = col_y + 45
    for key, val in ticket_details:
        draw.text((25, y_offset), f"{key}:", fill=(100, 116, 139), font=font_content)
        # Highlight values in bold or red
        fill_color = (239, 68, 68) if key == "Payment Status" else (15, 23, 42)
        if key == "Fine Amount":
            draw.text((150, y_offset), val, fill=(15, 23, 42), font=get_font("arial", 12, bold=True))
        else:
            draw.text((150, y_offset), val, fill=fill_color, font=font_content_bold)
        y_offset += 32
        
    # Right Column: Owner & RTO Details
    rx = canvas_w - 15 - col_w
    draw.rectangle([rx, col_y, rx + col_w, col_y + 220], outline=(226, 232, 240), width=1)
    # Right Header
    draw.rectangle([rx, col_y, rx + col_w, col_y + 30], fill=(241, 245, 249))
    draw.text((rx + 10, col_y + 8), "VEHICLE REGISTRY DATA (RTO)", fill=(15, 23, 42), font=font_section)
    
    rto_details = [
        ("Owner Name", rto_info.get("owner_name", "N/A")),
        ("Vehicle Make/Model", rto_info.get("vehicle_model", "N/A")),
        ("Fuel Type", rto_info.get("fuel_type", "N/A")),
        ("Insurance Status", rto_info.get("insurance_status", "N/A")),
        ("Registration Date", rto_info.get("registration_date", "N/A"))
    ]
    
    y_offset = col_y + 45
    for key, val in rto_details:
        draw.text((rx + 10, y_offset), f"{key}:", fill=(100, 116, 139), font=font_content)
        fill_color = (22, 163, 74) if "Active" in val else ((239, 68, 68) if val == "Expired" else (15, 23, 42))
        draw.text((rx + 150, y_offset), val, fill=fill_color, font=font_content_bold)
        y_offset += 32
        
    # 5. Visual Evidence Section
    evidence_y = col_y + 240
    evidence_h = 240
    draw.rectangle([15, evidence_y, canvas_w - 16, evidence_y + evidence_h], outline=(226, 232, 240), width=1)
    # Section Header
    draw.rectangle([15, evidence_y, canvas_w - 16, evidence_y + 30], fill=(241, 245, 249))
    draw.text((25, evidence_y + 8), "CERTIFIED VISUAL EVIDENCE LOGS", fill=(15, 23, 42), font=font_section)
    
    # Left Evidence Crop: Rider Crop (Resized)
    # Convert OpenCV numpy BGR arrays to PIL RGB
    if rider_crop is not None and rider_crop.size > 0:
        rider_rgb = cv2.cvtColor(rider_crop, cv2.COLOR_BGR2RGB)
        rider_pil = Image.fromarray(rider_rgb)
        # Resize maintaining aspect ratio or force fixed size
        rider_pil = rider_pil.resize((180, 180), Image.Resampling.LANCZOS)
        # Paste into canvas
        img.paste(rider_pil, (100, evidence_y + 45))
        # Draw frame around it
        draw.rectangle([99, evidence_y + 44, 281, evidence_y + 226], outline=(200, 200, 200), width=1)
        draw.text((100, evidence_y + 228), "1. Offender (No Helmet Headshot)", fill=(71, 85, 105), font=font_subtitle)
    else:
        # Draw placeholder
        draw.rectangle([99, evidence_y + 44, 281, evidence_y + 226], fill=(240, 240, 240), outline=(200, 200, 200))
        draw.text((120, evidence_y + 120), "RIDER EVIDENCE MISSING", fill=(150, 150, 150), font=font_subtitle)
        
    # Right Evidence Crop: Plate Crop (Resized)
    if plate_crop is not None and plate_crop.size > 0:
        plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
        plate_pil = Image.fromarray(plate_rgb)
        # Plate is horizontal, resize accordingly
        plate_pil = plate_pil.resize((300, 100), Image.Resampling.LANCZOS)
        # Paste plate crop
        img.paste(plate_pil, (400, evidence_y + 80))
        # Draw frame
        draw.rectangle([399, evidence_y + 79, 701, evidence_y + 181], outline=(200, 200, 200), width=1)
        draw.text((400, evidence_y + 185), "2. Registered License Plate Crop", fill=(71, 85, 105), font=font_subtitle)
    else:
        # Draw placeholder
        draw.rectangle([399, evidence_y + 79, 701, evidence_y + 181], fill=(240, 240, 240), outline=(200, 200, 200))
        draw.text((430, evidence_y + 120), "LICENSE PLATE CROP MISSING", fill=(150, 150, 150), font=font_subtitle)
        
    # 6. Footer section (Barcode, Info note, Official Signature)
    footer_y = evidence_y + evidence_h + 15
    draw_barcode(draw, 30, footer_y, width=180, height=45)
    font_barcode_text = get_font("arial", 9, bold=False)
    draw.text((30, footer_y + 50), f"* {violation_id.upper()} *", fill=(71, 85, 105), font=font_barcode_text)
    
    # Official Seal / Text
    font_disclaimer = get_font("arial", 9, bold=False)
    disclaimer_text = (
        "This is a system-generated document based on AI traffic monitoring camera evidence.\n"
        "Please scan the barcode or visit transport.gov.in/e-challan to settle the dues.\n"
        "Failure to clear the fine within 15 days will result in court summon."
    )
    draw.text((250, footer_y + 5), disclaimer_text, fill=(100, 116, 139), font=font_disclaimer)
    
    # Signature line
    draw.line([600, footer_y + 35, 750, footer_y + 35], fill=(150, 150, 150), width=1)
    font_sig = get_font("arial", 10, bold=True)
    draw.text((610, footer_y + 40), "AUTHORIZED SIGNATURE", fill=(100, 116, 139), font=font_sig)
    
    # Save Image
    challan_filename = f"challan_{violation_id}.png"
    challan_path = os.path.join(output_dir, challan_filename)
    img.save(challan_path)
    
    return challan_path
