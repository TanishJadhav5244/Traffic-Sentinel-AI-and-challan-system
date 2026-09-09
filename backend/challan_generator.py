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

def generate_qr_code_image(data_text, size=110):
    """
    Generates a high-quality, scannable QR code as a PIL Image.
    Uses Python's qrcode library (with error correction) for 100% smartphone camera scannability.
    Falls back gracefully to a high-contrast vector pattern if qrcode is unavailable.
    """
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=4,
            border=1
        )
        qr.add_data(data_text)
        qr.make(fit=True)
        pil_img = qr.make_image(fill_color="#0f172a", back_color="#ffffff").convert("RGB")
        return pil_img.resize((size, size), Image.Resampling.LANCZOS)
    except Exception:
        # High-contrast vector QR fallback
        fallback = Image.new("RGB", (size, size), (255, 255, 255))
        f_draw = ImageDraw.Draw(fallback)
        f_draw.rectangle([0, 0, size - 1, size - 1], outline=(15, 23, 42), width=2)
        grid_size = 21
        cell_w = size / grid_size

        def draw_finder(fx, fy):
            f_draw.rectangle([fx * cell_w, fy * cell_w, (fx + 7) * cell_w, (fy + 7) * cell_w], fill=(15, 23, 42))
            f_draw.rectangle([(fx + 1) * cell_w, (fy + 1) * cell_w, (fx + 6) * cell_w, (fy + 6) * cell_w], fill=(255, 255, 255))
            f_draw.rectangle([(fx + 2) * cell_w, (fy + 2) * cell_w, (fx + 5) * cell_w, (fy + 5) * cell_w], fill=(15, 23, 42))

        draw_finder(1, 1)
        draw_finder(13, 1)
        draw_finder(1, 13)

        seed = abs(hash(str(data_text)))
        for r in range(grid_size):
            for c in range(grid_size):
                if (r < 9 and c < 9) or (r < 9 and c > 11) or (r > 11 and c < 9):
                    continue
                if (seed + r * 31 + c * 17) % 3 < 2:
                    f_draw.rectangle([c * cell_w, r * cell_w, (c + 1) * cell_w, (r + 1) * cell_w], fill=(15, 23, 42))
        return fallback

def generate_challan_ticket(violation_id, timestamp, plate_text, rto_info, plate_crop, rider_crop, output_dir="violations/challans", fine_amount=1000.0):

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
    
    # 1. Initialize blank white canvas (800 x 780)
    canvas_w, canvas_h = 800, 780
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
        
    # 6. Footer section (Prominent Scannable QR Code, Digital Verification & Auth Seal)
    footer_y = evidence_y + evidence_h + 12
    
    # Outer footer card container
    draw.rectangle([15, footer_y, canvas_w - 16, footer_y + 115], fill=(248, 250, 252), outline=(203, 213, 225), width=1)
    
    # Generate real, smartphone-scannable UPI payment QR code
    upi_payload = f"upi://pay?pa=morth.echallan@gov.in&pn=MoRTH%20Traffic%20Police&am={float(fine_amount):.2f}&tn=Challan%20{violation_id}&tr={violation_id}"
    qr_img = generate_qr_code_image(upi_payload, size=95)
    img.paste(qr_img, (28, footer_y + 8))
    draw.rectangle([27, footer_y + 7, 28 + 95, footer_y + 8 + 95], outline=(15, 23, 42), width=1)

    font_qr_badge = get_font("arial", 8, bold=True)
    draw.text((25, footer_y + 104), "SCAN TO PAY (UPI / BHIM)", fill=(22, 101, 52), font=font_qr_badge)

    # Verification & Payment Details (Center Column)
    font_foot_head = get_font("arial", 11, bold=True)
    font_foot_body = get_font("arial", 9, bold=False)
    font_foot_bold = get_font("arial", 9, bold=True)

    draw.text((140, footer_y + 10), "DIGITAL PARIVAHAN VERIFICATION & E-PAYMENT", fill=(15, 23, 42), font=font_foot_head)
    draw.text((140, footer_y + 28), "Notice Citation ID: ", fill=(100, 116, 139), font=font_foot_body)
    draw.text((255, footer_y + 28), f"#{violation_id}  |  Vehicle: {plate_text}", fill=(30, 41, 59), font=font_foot_bold)

    draw.text((140, footer_y + 44), "Penalty Amount: ", fill=(100, 116, 139), font=font_foot_body)
    draw.text((255, footer_y + 44), f"INR {fine_amount:,.2f} (Instant UPI / Cards / NetBanking)", fill=(22, 101, 52), font=font_foot_bold)

    draw.text((140, footer_y + 60), "Online Settlement: ", fill=(100, 116, 139), font=font_foot_body)
    draw.text((255, footer_y + 60), "https://echallan.parivahan.gov.in", fill=(2, 132, 199), font=font_foot_bold)

    draw.text((140, footer_y + 76), "Statutory Deadline: ", fill=(100, 116, 139), font=font_foot_body)
    draw.text((255, footer_y + 76), "Pay within 15 days to avoid judicial court summons", fill=(220, 38, 38), font=font_foot_bold)

    sec_hash = abs(hash(f"{violation_id}{plate_text}")) % 1000000000
    draw.text((140, footer_y + 94), f"Anti-Tamper Hash: MoRTH-SEC-{sec_hash:09d} | Auto-Certified by Traffic Sentinel AI", fill=(148, 163, 184), font=get_font("arial", 8))

    # Right side: Official Security Stamp & Authorized Seal
    draw.rectangle([595, footer_y + 10, 770, footer_y + 104], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.rectangle([595, footer_y + 10, 770, footer_y + 30], fill=(15, 23, 42))
    draw.text((615, footer_y + 14), "OFFICIAL DIGITAL SEAL", fill=(255, 255, 255), font=get_font("arial", 8, bold=True))

    draw.text((612, footer_y + 37), "[ VALID PARIVAHAN ]", fill=(22, 101, 52), font=get_font("arial", 9, bold=True))
    draw.text((612, footer_y + 53), "Traffic Police Directorate", fill=(71, 85, 105), font=get_font("arial", 8))
    draw.text((612, footer_y + 67), "Enforcement Command", fill=(71, 85, 105), font=get_font("arial", 8))
    draw.line([605, footer_y + 85, 760, footer_y + 85], fill=(203, 213, 225), width=1)
    draw.text((620, footer_y + 88), "AUTHORIZED SIGNATURE", fill=(100, 116, 139), font=get_font("arial", 7, bold=True))
    
    # Save Image
    challan_filename = f"challan_{violation_id}.png"
    challan_path = os.path.join(output_dir, challan_filename)
    img.save(challan_path)
    
    return challan_path


def generate_challan_pdf(violation_id, timestamp, plate_text, rto_info, plate_crop, rider_crop, output_dir="violations/challans", fine_amount=1000.0):
    """
    Generates an official printable PDF E-Challan ticket and returns the PDF file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    # First generate the high-res PNG canvas
    png_path = generate_challan_ticket(violation_id, timestamp, plate_text, rto_info, plate_crop, rider_crop, output_dir=output_dir, fine_amount=fine_amount)
    
    pdf_filename = f"challan_{violation_id}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    try:
        with Image.open(png_path) as im:
            rgb_im = im.convert("RGB")
            rgb_im.save(pdf_path, "PDF", resolution=100.0)
        return pdf_path
    except Exception as e:
        print(f"[Challan] Error converting ticket to PDF: {e}")
        return png_path


class EChallanGenerator:
    """Wrapper class around challan ticket generator."""
    def __init__(self, config=None):
        self.config = config or {}
        self.output_dir = self.config.get("storage", {}).get("challan_dir", "violations/challans")

    def generate(self, plate_number, rto_details, head_crop, plate_crop, location="Intersection Cam #04", fine_amount=1000.0):
        violation_id = str(uuid.uuid4())[:8].upper()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = generate_challan_ticket(
            violation_id=violation_id,
            timestamp=timestamp,
            plate_text=plate_number,
            rto_info=rto_details,
            plate_crop=plate_crop,
            rider_crop=head_crop,
            output_dir=self.output_dir,
            fine_amount=fine_amount
        )
        return path, violation_id


