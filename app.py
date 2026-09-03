import os
import re
import cv2
import yaml
import time
import datetime
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

if not st.runtime.exists():
    import sys
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

from PIL import Image

from backend.detector import VehicleHelmetDetector
from backend.ocr_engine import LicensePlateOCR
from backend.db_helper import ViolationDatabase
from backend.rto_helper import query_rto
from backend.tracker import VehicleTracker
from backend.challan_payment import EChallanPaymentGateway
from backend.notifier import NotificationCenter
from frontend.styles import inject_custom_styles
from frontend.components import (
    render_gov_header,
    render_hero,
    render_section_header,
    content_panel,
    render_metric_cards,
    render_empty_state,
    render_quick_plate_chips,
    render_pipeline_steps,
    render_cctv_hud_header,
    render_telemetry_badge,
    render_status_chip,
)

# Shim for legacy panel helpers used in this file
def render_panel_start(css_class: str = ""):
    """Opens a visually bordered panel block."""
    cls = f' class="{css_class}"' if css_class else ""
    st.markdown(f'<div{cls}>', unsafe_allow_html=True)

def render_panel_end():
    """Closes the panel block opened by render_panel_start."""
    st.markdown('</div>', unsafe_allow_html=True)

# Set Streamlit page layout
st.set_page_config(
    page_title="Traffic Sentinel - Helmet & Plate Detection System",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_styles()

# Load configuration
@st.cache_resource
def load_config():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        return {
            "models": {"detector_weights": "models/best_detector.pt"},
            "ocr": {"default_engine": "easyocr", "tesseract_path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"},
            "storage": {"csv_log_path": "violations/violations_log.csv", "crop_dir": "violations/crops"}
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize core system helpers
@st.cache_resource
def get_detector(_config):
    return VehicleHelmetDetector(_config)

@st.cache_resource
def get_ocr_engine(_config):
    return LicensePlateOCR(_config)

@st.cache_resource
def get_db(_config):
    storage = _config.get("storage", {})
    return ViolationDatabase(
        csv_log_path=storage.get("csv_log_path", "violations/violations_log.csv"),
        crop_dir=storage.get("crop_dir", "violations/crops"),
        config=_config
    )

@st.cache_resource
def get_tracker():
    return VehicleTracker(pixels_per_meter=20.0, cooldown_seconds=20)

@st.cache_resource
def get_payment_gateway(_db, _config):
    return EChallanPaymentGateway(db_helper=_db, config=_config)

@st.cache_resource
def get_notification_center(_config):
    return NotificationCenter(_config)

detector = get_detector(config)
ocr_engine = get_ocr_engine(config)
db = get_db(config)
tracker = get_tracker()
payment_gateway = get_payment_gateway(db, config)
notification_center = get_notification_center(config)

# Official Government Top Header & Hero banner with live stats
render_gov_header()
_violation_df = db.get_all_violations()
render_hero(violation_count=len(_violation_df))

# ── Sidebar ──────────────────────────────────────────────
# System status at the top so the user sees it immediately
st.sidebar.markdown(
    '<div class="sidebar-status sidebar-status-live">'
    '● System Active'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Settings")

# ── OCR Engine ──
ocr_selection = st.sidebar.selectbox(
    "Text Recognition Engine",
    options=["easyocr", "tesseract"],
    index=0 if config.get("ocr", {}).get("default_engine") == "easyocr" else 1,
    help="EasyOCR is more accurate for most plates. Tesseract is faster but less reliable.",
)

# ── Detection Sensitivity (collapsible — advanced) ──
with st.sidebar.expander("Detection Sensitivity", expanded=False):
    st.caption("Adjust how strictly the AI detects objects. "
               "Lower values catch more but may include false positives.")
    rider_conf = st.slider(
        "Rider detection",
        0.10, 1.00,
        float(config.get("models", {}).get("confidence", {}).get("rider", 0.35)),
        key="slider_rider",
        help="Minimum confidence to detect a two-wheeler rider.",
    )
    helmet_conf = st.slider(
        "Helmet detection",
        0.10, 1.00,
        float(config.get("models", {}).get("confidence", {}).get("helmet", 0.40)),
        key="slider_helmet",
        help="Minimum confidence to classify helmet / no-helmet.",
    )
    plate_conf = st.slider(
        "Number plate detection",
        0.10, 1.00,
        float(config.get("models", {}).get("confidence", {}).get("license_plate", 0.40)),
        key="slider_plate",
        help="Minimum confidence to detect a license plate region.",
    )

# Set runtime settings back into classes
detector.thresholds["rider"] = rider_conf
detector.thresholds["helmet"] = helmet_conf
detector.thresholds["license_plate"] = plate_conf

# ── Speed Enforcement Radar ──
with st.sidebar.expander("⚡ Speed Enforcement Radar", expanded=False):
    st.caption("Configure speed radar thresholds and speed limit alerts.")
    speed_limit_kmh = st.slider(
        "Speed Limit (km/h)",
        20, 120, 60,
        step=5,
        key="slider_speed_limit",
        help="Vehicles moving faster than this speed threshold trigger automated over-speeding citations.",
    )

# ── Night / Low-Light Mode (collapsible) ──
with st.sidebar.expander("Low-Light / Night Mode", expanded=False):
    st.caption("Enhance dark or night-time footage for better plate reading.")
    enable_enhancer = st.checkbox(
        "Auto-enhance dark images",
        value=bool(config.get("ocr", {}).get("enhancer", {}).get("auto_low_light", True)),
        help="Automatically brightens dark plate crops before reading text.",
    )
    force_enhancer = st.checkbox(
        "Always apply night filter",
        value=bool(config.get("ocr", {}).get("enhancer", {}).get("force_enhancement", False)),
        help="Force brightness enhancement on every plate, even well-lit ones.",
    )
ocr_engine.enhancer_settings["auto_low_light"] = enable_enhancer
ocr_engine.enhancer_settings["force_enhancement"] = force_enhancer

# ── Quick Actions ──
st.sidebar.markdown("---")
sb_col1, sb_col2 = st.sidebar.columns(2)
with sb_col1:
    if st.button("🔄 Refresh", key="sb_btn_refresh", use_container_width=True, type="secondary"):
        st.cache_resource.clear()
        st.rerun()
with sb_col2:
    if st.button("🗑️ Reset", key="sb_btn_reset", use_container_width=True, type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# Tab setup
tab_detector, tab_database, tab_payment, tab_ocr_playground, tab_rto, tab_analytics, tab_rest_api = st.tabs([
    "📸 Live Violation Detector",
    "📊 Violations Database",
    "💳 E-Challan Payment Portal",
    "🔬 OCR Preprocessing Lab",
    "🔍 RTO Vehicle Registry",
    "📈 Analytics Dashboard",
    "🔌 REST API & Edge Hub",
])

# ---------------------------------------------------------
# TAB 1: Live Violation Detector
# ---------------------------------------------------------
with tab_detector:
    render_pipeline_steps(current_step=0)
    col_input, col_output = st.columns([1, 2], gap="large")

    with col_input:
        with content_panel("input"):
            render_section_header("Feed Upload", "Select a sample or upload your own image/video", "📤")
            upload_type = st.radio(
                "Input Source",
                ["Sample Assets", "Upload File (Image/Video)", "📷 Live Webcam Feed", "📹 24/7 CCTV Live Feed Simulator"],
                horizontal=True,
                label_visibility="collapsed",
            )
        
        input_image = None
        input_video_path = None
        cctv_cam_name = "CAM-01"
        cctv_cam_location = "MG Road Crossing"
        
        if upload_type == "📷 Live Webcam Feed":
            st.markdown("#### 📷 Live Camera Feed Capture")
            webcam_img = st.camera_input("Capture frame from connected webcam")
            if webcam_img is not None:
                file_bytes = np.asarray(bytearray(webcam_img.read()), dtype=np.uint8)
                input_image = cv2.imdecode(file_bytes, 1)

        elif upload_type == "📹 24/7 CCTV Live Feed Simulator":
            st.markdown("#### 📹 CCTV Surveillance Feeds")
            cctv_selection = st.selectbox(
                "Select Live Traffic Camera",
                options=[
                    "CAM-01: MG Road Crossing (North Sector)",
                    "CAM-02: Express Highway Toll Plaza (South)",
                    "CAM-03: City Center Square (West Corridor)"
                ]
            )
            cctv_cam_name = cctv_selection.split(":")[0]
            cctv_cam_location = cctv_selection.split("(")[0].split(":")[1].strip()
            
            # Map camera to sample video/image
            sample_dir = "test_assets"
            video_samples = [f for f in os.listdir(sample_dir) if f.endswith('.mp4')] if os.path.exists(sample_dir) else []
            if video_samples:
                input_video_path = os.path.join(sample_dir, video_samples[0])
            else:
                input_image = cv2.imread(os.path.join(sample_dir, "traffic_sample.png")) if os.path.exists(os.path.join(sample_dir, "traffic_sample.png")) else None

        elif upload_type == "Sample Assets":
            # List sample files
            sample_dir = "test_assets"
            os.makedirs(sample_dir, exist_ok=True)
            samples = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.mp4'))]
            
            if not samples:
                render_empty_state(
                    "No sample assets found",
                    "Generate sample images and configure model weights to get started",
                    "🖼️",
                )
                if st.button("⚡ Generate Sample Assets", type="secondary"):
                    with st.spinner("Generating sample images and configuring weights..."):
                        os.system("python backend/models/download_weights.py")
                        st.rerun()
            else:
                if "active_sample" not in st.session_state:
                    st.session_state.active_sample = samples[0]
                
                st.caption("⚡ **1-Click Quick Presets**:")
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    if st.button("🚦 Traffic", key="preset_traffic", use_container_width=True, type="secondary"):
                        for s in samples:
                            if "traffic" in s.lower():
                                st.session_state.active_sample = s
                                break
                with pcol2:
                    if st.button("📄 Clean", key="preset_clean", use_container_width=True, type="secondary"):
                        for s in samples:
                            if "clean" in s.lower():
                                st.session_state.active_sample = s
                                break
                with pcol3:
                    if st.button("🌧️ Skewed", key="preset_skewed", use_container_width=True, type="secondary"):
                        for s in samples:
                            if "skewed" in s.lower() or "noisy" in s.lower():
                                st.session_state.active_sample = s
                                break
                with pcol4:
                    if st.button("🎥 Video", key="preset_video", use_container_width=True, type="secondary"):
                        for s in samples:
                            if s.endswith(".mp4"):
                                st.session_state.active_sample = s
                                break

                idx = samples.index(st.session_state.active_sample) if st.session_state.active_sample in samples else 0
                selected_sample = st.selectbox("Select Sample Asset", samples, index=idx)
                sample_path = os.path.join(sample_dir, selected_sample)
                if sample_path.endswith('.mp4'):
                    input_video_path = sample_path
                else:
                    input_image = cv2.imread(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload Image or Video (MP4)", type=["jpg", "jpeg", "png", "mp4"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.mp4'):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    input_video_path = tfile.name
                else:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    input_image = cv2.imdecode(file_bytes, 1)

        # Initialize video processing state
        if "video_processing_active" not in st.session_state:
            st.session_state.video_processing_active = False

        def stop_video_processing():
            st.session_state.video_processing_active = False

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        
        # Conditional button rendering based on input type and running state
        is_video_input = (input_video_path is not None)
        
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            if is_video_input and st.session_state.video_processing_active:
                st.button("⏹ Stop Processing", type="secondary", use_container_width=True, on_click=stop_video_processing)
                run_pipeline = False
            else:
                run_pipeline = st.button("▶ Run System Inference", type="primary", use_container_width=True)
                if run_pipeline and is_video_input:
                    st.session_state.video_processing_active = True
        with btn_col2:
            if st.button("🔄 Reset", key="btn_reset_detector", use_container_width=True, type="secondary"):
                st.session_state.video_processing_active = False
                st.rerun()

    with col_output:
        with content_panel("output"):
            render_section_header("System Visual Output", "Detection results and violation evidence appear here", "🎯")
            if upload_type == "📹 24/7 CCTV Live Feed Simulator":
                render_cctv_hud_header(cam_id=cctv_cam_name, location=cctv_cam_location)
        
        if run_pipeline or (is_video_input and st.session_state.video_processing_active):
            # PROCESS IMAGE
            if input_image is not None:
                st.info("Processing image pipeline with multi-violation AI inference...")
                img_to_proc = input_image.copy()
                
                # Step 1: Detect
                detections = detector.detect(img_to_proc)
                
                # Step 2: Multi-violation association
                violations = detector.associate_violations(detections, speed_kmh=0.0, speed_limit=speed_limit_kmh)
                
                # Step 3: OCR on associated plate regions
                logged_violations = []
                for v in violations:
                    p_box = v["plate"]["box"]
                    r_box = v["rider"]["box"]
                    
                    # Crop image arrays with 8% dynamic padding margin to prevent character truncation
                    h, w = img_to_proc.shape[:2]
                    pw_pad = int((p_box[2] - p_box[0]) * 0.08)
                    ph_pad = int((p_box[3] - p_box[1]) * 0.08)
                    px1, py1 = max(0, p_box[0] - pw_pad), max(0, p_box[1] - ph_pad)
                    px2, py2 = min(w, p_box[2] + pw_pad), min(h, p_box[3] + ph_pad)
                    
                    rx1, ry1, rx2, ry2 = max(0, r_box[0]), max(0, r_box[1]), min(w, r_box[2]), min(h, r_box[3])
                    
                    plate_crop = img_to_proc[py1:py2, px1:px2]
                    rider_crop = img_to_proc[ry1:ry2, rx1:rx2]
                    
                    # Run OCR
                    ocr_res = ocr_engine.recognize(
                        plate_crop, engine=ocr_selection,
                        enable_enhancer=enable_enhancer, force_enhancer=force_enhancer
                    )
                    plate_text = ocr_res["cleaned_text"] or "UNKNOWN"
                    ocr_conf = ocr_res["confidence"]
                    was_low_light = ocr_res.get("was_low_light", False)
                    
                    v_type_str = " + ".join(v.get("violation_types", ["No Helmet"]))
                    fine_amt = v.get("fine_amount", 1000.0)

                    # Save and Log to DB
                    record = db.log_violation(
                        frame_timestamp="00:00:00",
                        plate_crop=plate_crop,
                        rider_crop=rider_crop,
                        plate_text=plate_text,
                        ocr_conf=ocr_conf,
                        helmet_status="no-helmet" if "No Helmet" in v_type_str else "compliant",
                        night_mode=was_low_light,
                        violation_type=v_type_str,
                        speed_recorded=0.0,
                        camera_id=cctv_cam_name,
                        location=cctv_cam_location,
                        challan_amount=fine_amt
                    )
                    logged_violations.append((record, plate_crop, rider_crop, ocr_res, v))
                    
                # Annotate and show image
                annotated_img = detector.draw_annotations(img_to_proc, detections, violations, speed_limit=speed_limit_kmh)
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                
                # Show cards below
                if logged_violations:
                    st.error(f"🚨 Flagged {len(logged_violations)} Traffic Violations!")
                    card_cols = st.columns(min(3, len(logged_violations)))
                    for idx, (rec, p_crp, r_crp, ocr_info, v_info) in enumerate(logged_violations):
                        with card_cols[idx % 3]:
                            night_badge = '<span style="background:rgba(251,191,36,0.2);color:#fbbf24;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;margin-left:6px;">🌙 Night Boost</span>' if rec.get('night_mode') or ocr_info.get('was_low_light') else ''
                            v_badge_color = "#ef4444" if "Triple" in rec.get('violation_type', '') else "#f59e0b"
                            st.markdown(f"""
                            <div class="violation-card">
                                <div class="violation-title">Violation ID: {rec['violation_id']} {night_badge}</div>
                                <div style="margin: 4px 0 8px 0;">
                                    <span style="background:rgba(239,68,68,0.2);color:{v_badge_color};padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;">⚠️ {rec.get('violation_type', 'No Helmet')}</span>
                                    <span style="background:rgba(34,197,94,0.2);color:#22c55e;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;margin-left:4px;">₹{rec.get('challan_amount', 1000):,.0f} Fine</span>
                                </div>
                                <b>Plate:</b> <span class="metric-value">{rec['plate_text']}</span><br>
                                <b>Owner:</b> {rec.get('owner_name', 'Unknown')}<br>
                                <b>Vehicle:</b> {rec.get('vehicle_model', 'Unknown')}<br>
                                <b>Camera:</b> {rec.get('camera_id', 'CAM-01')} ({rec.get('location', 'Surveillance Zone')})<br>
                                <b>Time:</b> {rec['timestamp']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show side-by-side crops
                            crop_c1, crop_c2 = st.columns(2)
                            with crop_c1:
                                if r_crp.size > 0:
                                    st.image(cv2.cvtColor(r_crp, cv2.COLOR_BGR2RGB), caption="Rider / Group", use_container_width=True)
                            with crop_c2:
                                if p_crp.size > 0:
                                    st.image(cv2.cvtColor(p_crp, cv2.COLOR_BGR2RGB), caption="Plate", use_container_width=True)
                                    
                            # Download button for E-Challan
                            c_path = rec.get("challan_path")
                            if c_path and os.path.exists(c_path):
                                with open(c_path, "rb") as f:
                                    st.download_button(
                                        label="📄 Download E-Challan",
                                        data=f,
                                        file_name=f"challan_{rec['violation_id']}.png",
                                        mime="image/png",
                                        key=f"dl_{rec['violation_id']}_{idx}",
                                        use_container_width=True
                                    )
                else:
                    st.success("✅ No Traffic Violations detected. Compliant stream.")
                    
            # PROCESS VIDEO / CCTV STREAM
            elif input_video_path is not None:
                st.info(f"Processing CCTV Video Stream [{cctv_cam_name}] with Vehicle Tracking & Radar...")
                video_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    st.error("Error opening video file.")
                    st.session_state.video_processing_active = False
                else:
                    try:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        
                        # De-duplication set to prevent spamming CSV logs for same plate within video
                        recent_detections = {}
                        
                        frame_idx = 0
                        violations_found = 0
                        
                        while cap.isOpened() and st.session_state.video_processing_active:
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            frame_idx += 1
                            
                            # Process every 3rd frame to ensure high fidelity and smooth tracking on CPU
                            if frame_idx % 3 != 0:
                                continue
                                
                            timestamp_str = str(datetime.timedelta(seconds=int(frame_idx / fps)))
                            progress_val = min(1.0, max(0.0, frame_idx / frame_count))
                            progress_bar.progress(progress_val)
                            status_text.text(f"Camera: {cctv_cam_name} | Frame: {frame_idx}/{frame_count} ({progress_val*100:.1f}%) | Speed Limit: {speed_limit_kmh} km/h | Time: {timestamp_str}")
                            
                            # Pipeline Inference
                            detections = detector.detect(frame)
                            
                            # Update Vehicle Tracker
                            vehicle_boxes = [r["box"] for r in detections.get("riders", [])]
                            assigned_tracks = tracker.update(vehicle_boxes)
                            active_tracks = tracker.get_all_active_tracks(assigned_tracks, speed_limit=speed_limit_kmh)
                            
                            # Multi-violation association
                            violations = detector.associate_violations(detections, speed_limit=speed_limit_kmh)
                            
                            current_frame_violations = []
                            for v in violations:
                                p_box = v["plate"]["box"]
                                r_box = v["rider"]["box"]
                                
                                # Match rider to track to read estimated speed
                                r_center = ((r_box[0] + r_box[2]) / 2.0, (r_box[1] + r_box[3]) / 2.0)
                                matched_speed = 0.0
                                for tid, tinfo in active_tracks.items():
                                    tc = tinfo["centroid"]
                                    if np.hypot(tc[0] - r_center[0], tc[1] - r_center[1]) < 90:
                                        matched_speed = tinfo["speed"]
                                        break
                                
                                # If speed exceeds limit, append speed violation
                                if matched_speed > speed_limit_kmh and not any("Speed" in vt for vt in v["violation_types"]):
                                    v["violation_types"].append(f"Over-Speeding ({matched_speed:.0f} km/h)")
                                    v["fine_amount"] += 2000.0
                                
                                v["speed_kmh"] = matched_speed
                                
                                h, w = frame.shape[:2]
                                pw_pad = int((p_box[2] - p_box[0]) * 0.08)
                                ph_pad = int((p_box[3] - p_box[1]) * 0.08)
                                px1, py1 = max(0, p_box[0] - pw_pad), max(0, p_box[1] - ph_pad)
                                px2, py2 = min(w, p_box[2] + pw_pad), min(h, p_box[3] + ph_pad)
                                
                                rx1, ry1, rx2, ry2 = max(0, r_box[0]), max(0, r_box[1]), min(w, r_box[2]), min(h, r_box[3])
                                
                                plate_crop = frame[py1:py2, px1:px2]
                                rider_crop = frame[ry1:ry2, rx1:rx2]
                                
                                # Run OCR
                                ocr_res = ocr_engine.recognize(
                                    plate_crop, engine=ocr_selection,
                                    enable_enhancer=enable_enhancer, force_enhancer=force_enhancer
                                )
                                plate_text = ocr_res["cleaned_text"] or "UNKNOWN"
                                ocr_conf = ocr_res["confidence"]
                                was_low_light = ocr_res.get("was_low_light", False)
                                
                                # De-duplicate plate text within rolling 5-second window
                                current_time = frame_idx / fps
                                if plate_text != "UNKNOWN" and len(plate_text) >= 5:
                                    last_seen = recent_detections.get(plate_text, -999)
                                    if current_time - last_seen > 5.0:
                                        recent_detections[plate_text] = current_time
                                        v_type_str = " + ".join(v.get("violation_types", ["No Helmet"]))
                                        
                                        # Log to CSV
                                        db.log_violation(
                                            frame_timestamp=timestamp_str,
                                            plate_crop=plate_crop,
                                            rider_crop=rider_crop,
                                            plate_text=plate_text,
                                            ocr_conf=ocr_conf,
                                            helmet_status="no-helmet" if "No Helmet" in v_type_str else "compliant",
                                            night_mode=was_low_light,
                                            violation_type=v_type_str,
                                            speed_recorded=matched_speed,
                                            camera_id=cctv_cam_name,
                                            location=cctv_cam_location,
                                            challan_amount=v.get("fine_amount", 1000.0)
                                        )
                                        violations_found += 1
                                        st.toast(f"🚨 {v_type_str}: {plate_text} (Fine: ₹{v.get('fine_amount', 1000):,.0f})", icon="🚨")
                                        
                            # Draw bounding boxes and update video frame
                            annotated = detector.draw_annotations(frame, detections, violations, tracked_vehicles=active_tracks, speed_limit=speed_limit_kmh)
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(annotated_rgb, use_container_width=True)
                            
                            # Keep frame rate smooth
                            time.sleep(0.01)
                            
                        if not st.session_state.video_processing_active:
                            status_text.warning("Stream processing stopped by user.")
                        else:
                            st.session_state.video_processing_active = False
                            status_text.success(f"Stream Processing Complete! Identified & logged {violations_found} unique traffic violations.")
                    finally:
                        cap.release()
        else:
            render_empty_state(
                "Ready to scan",
                "Upload or select an asset, then click Run System Inference",
                "🔍",
            )

# ---------------------------------------------------------
# TAB 2: Violations Database
# ---------------------------------------------------------
with tab_database:
    render_section_header(
        "Flagged Traffic Violations Log",
        "Search, filter, and export all recorded helmet violations",
        "📊",
    )

    col_reload, col_spacer = st.columns([1, 4])
    with col_reload:
        if st.button("🔄 Reload Data", type="secondary", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    df = db.get_all_violations()

    if df.empty:
        render_empty_state(
            "No violations recorded",
            "Clean roads! Run the detector to log new violations.",
            "✅",
        )
    else:
        total_fines_calc = df["challan_amount"].sum() if "challan_amount" in df.columns else len(df) * 1000
        render_metric_cards([
            {"icon": "🚨", "label": "Violations Detected", "value": len(df)},
            {"icon": "💰", "label": "Total Fines Levied", "value": f"₹{total_fines_calc:,.0f}"},
            {"icon": "🏍️", "label": "Compliance Rate", "value": "93.8%", "delta": "+2.4% weekly"},
        ])
        st.markdown("<br>", unsafe_allow_html=True)
            
        # Group aggregations
        state_counts = pd.DataFrame()
        brand_counts = pd.DataFrame()
        
        if "plate_text" in df.columns:
            states = df["plate_text"].dropna().astype(str).str.upper().str[:2]
            states = states[states.str.match(r'^[A-Z]{2}$', na=False)]
            if not states.empty:
                state_counts = states.value_counts().reset_index()
                state_counts.columns = ["State", "Violations"]
                
        if "vehicle_model" in df.columns:
            brands = df["vehicle_model"].dropna().astype(str).apply(lambda x: x.split()[0] if x else "Unknown")
            if not brands.empty:
                brand_counts = brands.value_counts().reset_index()
                brand_counts.columns = ["Manufacturer", "Violations"]
                
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("**Violations by Registered State**")
            if not state_counts.empty and state_counts["Violations"].sum() > 0:
                st.bar_chart(state_counts.set_index("State"))
            else:
                st.caption("Not enough data to display state distribution.")
        with col_c2:
            st.write("**Violations by Vehicle Brand**")
            if not brand_counts.empty and brand_counts["Violations"].sum() > 0:
                st.bar_chart(brand_counts.set_index("Manufacturer"))
            else:
                st.caption("Not enough data to display manufacturer distribution.")
                
        st.divider()

        # Multi-Criteria Filter Bar
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([2, 1, 1, 1, 1])
        with f_col1:
            search_query = st.text_input("🔍 Search by Plate Number", "")
        with f_col2:
            violation_filter = st.selectbox("🚨 Violation Type", ["All Types", "No Helmet", "Triple Riding", "Over-Speeding"])
        with f_col3:
            night_filter = st.selectbox("🌙 Night-Vision Mode", ["All Feed Modes", "Night Enhanced Only", "Daylight Only"])
        with f_col4:
            available_states = ["All States"]
            if "plate_text" in df.columns:
                st_list = df["plate_text"].dropna().astype(str).str.upper().str[:2]
                st_list = sorted(list(set(st_list[st_list.str.match(r'^[A-Z]{2}$', na=False)])))
                available_states.extend(st_list)
            selected_state_filter = st.selectbox("🗺️ State Registry", available_states)
        with f_col5:
            status_filter = st.selectbox("📌 Challan Status", ["All Statuses", "Pending", "Paid", "Disputed"])

        df_filtered = df.copy()
        if search_query:
            df_filtered = df_filtered[df_filtered["plate_text"].astype(str).str.contains(search_query.upper(), na=False)]
        if violation_filter != "All Types" and "violation_type" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["violation_type"].astype(str).str.contains(violation_filter, case=False, na=False)]
        if night_filter == "Night Enhanced Only" and "night_mode" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["night_mode"] == True]
        elif night_filter == "Daylight Only" and "night_mode" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["night_mode"] == False]
        if selected_state_filter != "All States":
            df_filtered = df_filtered[df_filtered["plate_text"].astype(str).str.upper().str.startswith(selected_state_filter)]
        if status_filter != "All Statuses" and "status" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["status"].astype(str).str.capitalize() == status_filter.capitalize()]
            
        # Display table
        preferred_cols = ["violation_id", "timestamp", "violation_type", "plate_text", "speed_recorded", "camera_id", "location", "challan_amount", "owner_name", "status"]
        show_cols = [c for c in preferred_cols if c in df_filtered.columns]
        st.dataframe(
            df_filtered[show_cols],
            use_container_width=True
        )
        
        # Interactive Grid to see crops and challans
        st.write("#### Detailed Violation Evidence & Challan Tickets")
        for idx, row in df_filtered.iterrows():
            with st.container():
                col_meta, col_rider_crop, col_plate_crop, col_challan = st.columns([2, 1, 1, 1.8])
                with col_meta:
                    st.write(f"**Violation ID:** `{row['violation_id']}`")
                    st.write(f"**Timestamp:** {row['timestamp']}")
                    st.markdown(f"**Vehicle Plate:** <span class='metric-value'>{row['plate_text']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Violation:** <span style='background:rgba(239,68,68,0.2);color:#ef4444;padding:2px 8px;border-radius:10px;font-size:0.8rem;font-weight:700;'>{row.get('violation_type', 'No Helmet')}</span>", unsafe_allow_html=True)
                    
                    spd = row.get('speed_recorded', 0.0)
                    if spd and float(spd) > 0:
                        st.write(f"**Recorded Speed:** `{float(spd):.1f} km/h`")
                    st.write(f"**Camera / Location:** {row.get('camera_id', 'CAM-01')} — {row.get('location', 'MG Road Crossing')}")
                    st.write(f"**Owner:** {row.get('owner_name', 'Unknown')}")
                    st.write(f"**Vehicle Model:** {row.get('vehicle_model', 'Unknown')}")
                    st.write(f"**Fine Levied:** ₹{row.get('challan_amount', 1000.0):,.2f}")
                    
                    curr_status = str(row.get('status', 'Pending'))
                    st.markdown(f"**Status:** {render_status_chip(curr_status)}", unsafe_allow_html=True)
                    
                    st.caption("Update Status:")
                    btn_s1, btn_s2, btn_s3 = st.columns(3)
                    with btn_s1:
                        if st.button("🟢 Paid", key=f"btn_paid_{row['violation_id']}_{idx}", use_container_width=True):
                            db.update_violation_status(row['violation_id'], "Paid")
                            st.toast(f"Challan {row['violation_id']} marked as Paid!")
                            st.rerun()
                    with btn_s2:
                        if st.button("🔴 Dispute", key=f"btn_disp_{row['violation_id']}_{idx}", use_container_width=True):
                            db.update_violation_status(row['violation_id'], "Disputed")
                            st.toast(f"Challan {row['violation_id']} marked as Disputed!")
                            st.rerun()
                    with btn_s3:
                        if st.button("🟡 Pending", key=f"btn_rst_{row['violation_id']}_{idx}", use_container_width=True):
                            db.update_violation_status(row['violation_id'], "Pending")
                            st.toast(f"Challan {row['violation_id']} reset to Pending!")
                            st.rerun()
                            
                    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
                    if st.button("📩 Dispatch SMS / Email Notice", key=f"btn_notify_{row['violation_id']}_{idx}", use_container_width=True, type="secondary"):
                        with st.spinner("Dispatching notification via SMS & Email..."):
                            dispatch_res = notification_center.dispatch_challan_notice(row, channels=["email", "sms"])
                        ch = dispatch_res.get("channels", {})
                        email_info = ch.get("email", {})
                        sms_info = ch.get("sms", {})
                        st.toast(f"📩 Notice dispatched to {row.get('owner_name', 'Owner')}!", icon="📩")
                        st.markdown(f"""
                        <div style="background: rgba(15,23,42,0.9); border: 1px solid #3b82f6; border-radius: 10px; padding: 14px; margin-top: 8px; font-size: 0.85rem;">
                            <div style="font-weight:700; color:#60a5fa; margin-bottom:8px;">📩 Notification Dispatch Summary</div>
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
                                <div style="background:rgba(59,130,246,0.1); padding:10px; border-radius:8px; border:1px solid rgba(59,130,246,0.2);">
                                    <div style="color:#93c5fd; font-weight:600;">✉️ Email</div>
                                    <div>To: <code>{email_info.get('recipient', 'N/A')}</code></div>
                                    <div>Status: <span style="color:#22c55e;">{email_info.get('status', 'N/A')}</span></div>
                                </div>
                                <div style="background:rgba(34,197,94,0.1); padding:10px; border-radius:8px; border:1px solid rgba(34,197,94,0.2);">
                                    <div style="color:#86efac; font-weight:600;">📱 SMS</div>
                                    <div>To: <code>{sms_info.get('recipient', 'N/A')}</code></div>
                                    <div>Status: <span style="color:#22c55e;">{sms_info.get('status', 'N/A')}</span></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                with col_rider_crop:
                    r_path = row["rider_crop_path"]
                    if os.path.exists(str(r_path)):
                        st.image(r_path, caption="Rider Head Crop", use_container_width=True)
                    else:
                        st.caption("No Rider Crop Saved")
                with col_plate_crop:
                    p_path = row["plate_crop_path"]
                    if os.path.exists(str(p_path)):
                        st.image(p_path, caption="License Plate", use_container_width=True)
                    else:
                        st.caption("No Plate Crop Saved")
                with col_challan:
                    c_path = row.get("challan_path", "")
                    if pd.notna(c_path) and os.path.exists(str(c_path)):
                        st.image(str(c_path), caption="Challan Ticket Preview", use_container_width=True)
                        dl_c1, dl_c2 = st.columns(2)
                        with dl_c1:
                            with open(str(c_path), "rb") as f:
                                st.download_button(
                                    label="📥 PNG Ticket",
                                    data=f,
                                    file_name=f"challan_{row['violation_id']}.png",
                                    mime="image/png",
                                    key=f"db_dl_png_{row['violation_id']}_{idx}",
                                    use_container_width=True
                                )
                        with dl_c2:
                            pdf_file_path = str(c_path).replace(".png", ".pdf")
                            if not os.path.exists(pdf_file_path) and os.path.exists(str(c_path)):
                                try:
                                    with Image.open(str(c_path)) as im:
                                        im.convert("RGB").save(pdf_file_path, "PDF", resolution=100.0)
                                except Exception:
                                    pass
                            if os.path.exists(pdf_file_path):
                                with open(pdf_file_path, "rb") as f_pdf:
                                    st.download_button(
                                        label="📄 Official PDF",
                                        data=f_pdf,
                                        file_name=f"challan_{row['violation_id']}.pdf",
                                        mime="application/pdf",
                                        key=f"db_dl_pdf_{row['violation_id']}_{idx}",
                                        use_container_width=True
                                    )
                    else:
                        st.caption("No Challan Ticket Generated")
                st.divider()
                
        # Database actions
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            # Download CSV link
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Database as CSV",
                data=csv_data,
                file_name="helmet_violations_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_act2:
            if st.button("🗑️ Wipe Database Logs", type="secondary", use_container_width=True):
                db.clear_database()
                st.warning("All records and cropped image binaries have been deleted.")
                st.rerun()

# ---------------------------------------------------------
# TAB 3: E-Challan Citizen Payment & Tax Receipt Portal
# ---------------------------------------------------------
with tab_payment:
    render_section_header(
        "Citizen E-Challan & Digital Payment Gateway",
        "Search citations by license plate or challan ID, settle traffic fines via UPI QR / Razorpay, and download official MoRTH Tax Receipts.",
        "💳"
    )

    # ── Real-Time Fine Recovery & Payment Ledger KPI Metrics ──
    pay_analytics = payment_gateway.get_payment_analytics()
    kpi_p1, kpi_p2, kpi_p3, kpi_p4 = st.columns(4)
    with kpi_p1:
        st.metric("Total Revenue Collected", f"₹{pay_analytics.get('total_revenue_inr', 0.0):,.0f}", delta=f"{pay_analytics.get('total_paid_tickets', 0)} Settled")
    with kpi_p2:
        st.metric("Outstanding Unpaid Penalties", f"₹{pay_analytics.get('pending_revenue_inr', 0.0):,.0f}", delta=f"{pay_analytics.get('total_pending_tickets', 0)} Pending", delta_color="inverse")
    with kpi_p3:
        st.metric("Fine Recovery Rate", f"{pay_analytics.get('recovery_rate_pct', 0.0)}%", delta="MoRTH Target: 85%")
    with kpi_p4:
        method_counts = pay_analytics.get("method_breakdown", {})
        top_method = max(method_counts, key=method_counts.get) if method_counts else "UPI_ONLINE"
        st.metric("Preferred Channel", top_method.replace("_", " "), delta=f"{sum(method_counts.values())} txns")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    pay_quick_plates = ["MH12DE5678", "MH12AB1234", "DL3CAY1111", "MH10BM2431", "MH10ER9193"]
    if "pay_query_input" not in st.session_state:
        st.session_state.pay_query_input = ""

    render_quick_plate_chips(pay_quick_plates, key_prefix="btn_pay_chip")

    render_panel_start("panel-input")
    with st.form("challan_payment_search_form"):
        pay_search_val = st.text_input(
            "Enter License Plate Number or Challan ID (e.g. MH12DE5678, DL3CAY1111, 763c6f40)",
            value=st.session_state.pay_query_input,
            key="pay_search_text_input",
            placeholder="MH12DE5678"
        )
        btn_query_fines = st.form_submit_button("🔎 Search Outstanding Challans", use_container_width=True, type="primary")
    render_panel_end()

    all_v = db.get_all_violations()
    target_records = pd.DataFrame()

    if pay_search_val.strip():
        clean_q = pay_search_val.strip().upper().replace(" ", "")
        if not all_v.empty:
            match_plate = all_v["plate_text"].astype(str).str.upper().str.replace(" ", "").str.contains(clean_q, na=False)
            match_id = all_v["violation_id"].astype(str).str.upper().str.contains(clean_q, na=False)
            target_records = all_v[match_plate | match_id]

    if not pay_search_val.strip():
        # Show all pending citations by default
        if not all_v.empty and "status" in all_v.columns:
            target_records = all_v[all_v["status"].astype(str).str.upper() == "PENDING"]

    if target_records.empty and pay_search_val.strip():
        st.success(f"🎉 No outstanding unpaid challans found for `{pay_search_val.upper()}`! Zero traffic penalties pending.")
    elif target_records.empty and not pay_search_val.strip():
        render_empty_state("No Unpaid Citations", "All traffic citations in the system have been settled and cleared.", "✅")
    else:
        pending_count = (target_records["status"].astype(str).str.upper() == "PENDING").sum() if "status" in target_records.columns else len(target_records)
        st.info(f"📋 Found **{len(target_records)}** total citation records (**{pending_count} Pending Payment**).")

        for idx, row in target_records.iterrows():
            v_id = str(row["violation_id"])
            c_status = str(row.get("status", "Pending")).capitalize()
            amt = float(row.get("challan_amount", 1000.0))
            is_paid = (c_status.upper() == "PAID")

            with st.container():
                st.markdown(f"""
                <div style="background: rgba(15,23,42,0.85); border: 1px solid {'#22c55e' if is_paid else '#ef4444'}; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px; margin-bottom: 12px;">
                        <div>
                            <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">E-Challan #{v_id}</span>
                            <span style="margin-left: 8px; font-size: 0.8rem; color: #94a3b8;">Issued: {row['timestamp']}</span>
                        </div>
                        <div>
                            <span class="rc-badge {'rc-badge-active' if is_paid else 'rc-badge-expired'}">{c_status}</span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; font-size: 0.9rem;">
                        <div><span style="color:#94a3b8;">Vehicle Number:</span><br><b style="color:#f59e0b; font-size: 1.05rem;">{row['plate_text']}</b></div>
                        <div><span style="color:#94a3b8;">Registered Owner:</span><br><b>{row.get('owner_name', 'Unknown')}</b></div>
                        <div><span style="color:#94a3b8;">Violation Type:</span><br><b style="color:#ef4444;">{row.get('violation_type', 'No Helmet')}</b></div>
                        <div><span style="color:#94a3b8;">Location / Camera:</span><br><b>{row.get('location', 'Surveillance Sector')}</b></div>
                        <div><span style="color:#94a3b8;">Penalty Amount:</span><br><b style="color:#22c55e; font-size: 1.15rem;">₹{amt:,.2f}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if not is_paid:
                    col_pay_method, col_qr_code = st.columns([3, 2], gap="large")
                    with col_pay_method:
                        st.markdown("#### 💳 Choose Payment Mode")
                        pay_method = st.radio(
                            "Select Payment Gateway",
                            [
                                "UPI (Google Pay / PhonePe / Paytm / BHIM)",
                                "Debit / Credit Card (Visa, RuPay, MasterCard)",
                                "NetBanking (SBI, HDFC, ICICI, Axis, PNB)",
                                "Digital Wallet (Amazon Pay, Paytm Wallet)"
                            ],
                            key=f"pm_sel_{v_id}_{idx}"
                        )

                        clean_method_code = "UPI_ONLINE"
                        if "Card" in pay_method:
                            clean_method_code = "CARD_GATEWAY"
                        elif "NetBanking" in pay_method:
                            clean_method_code = "NETBANKING"
                        elif "Wallet" in pay_method:
                            clean_method_code = "WALLET"

                        if st.button(f"⚡ Settle Challan #{v_id} (₹{amt:,.0f}) Now", key=f"btn_execute_pay_{v_id}_{idx}", type="primary", use_container_width=True):
                            with st.spinner("Processing transaction via secure payment gateway..."):
                                res = payment_gateway.process_payment(v_id, payment_method=clean_method_code)
                                if res.get("status") == "SUCCESS":
                                    st.success(f"🎉 Payment Verified & Confirmed! Transaction ID: `{res.get('transaction_id')}`")
                                    # Auto-dispatch payment confirmation notification
                                    pay_notice_record = dict(row)
                                    pay_notice_record["status"] = "Paid"
                                    notification_center.dispatch_challan_notice(pay_notice_record, channels=["email", "sms"])
                                    st.info("📩 Payment confirmation dispatched via SMS & Email to registered owner.")
                                    st.toast(f"✅ Challan #{v_id} cleared! Receipt generated & notification sent.", icon="✅")
                                    st.rerun()
                                else:
                                    st.error(f"Payment error: {res.get('message', 'Failed to process')}")

                    with col_qr_code:
                        st.markdown("#### 📱 Scan UPI QR to Pay")
                        st.caption(f"Scan with any UPI app to pay **₹{amt:,.2f}** instantly to MoRTH.")
                        # UPI QR Simulation Box
                        st.markdown(f"""
                        <div style="background: #ffffff; padding: 12px; border-radius: 10px; width: 170px; text-align: center; border: 2px solid #22c55e; margin: 0 auto 10px auto;">
                            <img src="https://api.qrserver.com/v1/create-qr-code/?size=140x140&data=upi://pay?pa=morth.challan@gov.in&pn=MoRTH%20Traffic%20Police&am={amt:.2f}&tn=Challan%20{v_id}" width="140" height="140" style="display:block; margin:0 auto;"/>
                            <div style="color: #0f172a; font-size: 0.7rem; font-weight: 700; margin-top: 4px;">SCAN & PAY ₹{amt:.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Paid receipt actions
                    st.success(f"✔ **Challan Cleared**: This citation is marked as **PAID**. No further action required.")
                    # Check for receipt files (PNG and PDF)
                    receipts_dir = "violations/challans"
                    png_receipts = [f for f in os.listdir(receipts_dir) if v_id in f and f.startswith("receipt_") and f.endswith(".png")] if os.path.exists(receipts_dir) else []
                    pdf_receipts = [f for f in os.listdir(receipts_dir) if v_id in f and f.startswith("receipt_") and f.endswith(".pdf")] if os.path.exists(receipts_dir) else []

                    if png_receipts:
                        rcp_file = os.path.join(receipts_dir, png_receipts[0])
                        col_rcp_img, col_rcp_dl = st.columns([3, 1.2])
                        with col_rcp_img:
                            st.image(rcp_file, caption="Official MoRTH Payment Tax Receipt", use_container_width=True)
                        with col_rcp_dl:
                            with open(rcp_file, "rb") as rf:
                                st.download_button(
                                    label="📥 Tax Receipt (PNG)",
                                    data=rf,
                                    file_name=f"receipt_{v_id}.png",
                                    mime="image/png",
                                    key=f"dl_rcp_png_{v_id}_{idx}",
                                    use_container_width=True
                                )
                            if pdf_receipts:
                                pdf_file = os.path.join(receipts_dir, pdf_receipts[0])
                                with open(pdf_file, "rb") as pf:
                                    st.download_button(
                                        label="📄 Official PDF Receipt",
                                        data=pf,
                                        file_name=f"receipt_{v_id}.pdf",
                                        mime="application/pdf",
                                        key=f"dl_rcp_pdf_{v_id}_{idx}",
                                        use_container_width=True
                                    )
                st.divider()

    # ── Notification Dispatch History ──────────────────────────────
    st.markdown("---")
    render_section_header(
        "Notification Dispatch History",
        "Real-time SMS & Email delivery audit trail for all E-Challan notices dispatched through this session.",
        "📋"
    )

    notif_history = notification_center.get_notification_history()
    notif_stats = notification_center.get_stats()

    nstat_c1, nstat_c2, nstat_c3 = st.columns(3)
    with nstat_c1:
        st.metric("Total Dispatched", notif_stats.get("total_dispatched", 0))
    with nstat_c2:
        st.metric("Emails Sent", notif_stats.get("email_sent", 0))
    with nstat_c3:
        st.metric("SMS Sent", notif_stats.get("sms_sent", 0))

    if notif_history:
        for nidx, notif in enumerate(reversed(notif_history[-20:])):
            ch = notif.get("channels", {})
            email_ch = ch.get("email", {})
            sms_ch = ch.get("sms", {})
            st.markdown(f"""
            <div style="background: rgba(15,23,42,0.7); border: 1px solid rgba(59,130,246,0.3); border-radius: 8px; padding: 12px; margin-bottom: 8px; font-size: 0.82rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#60a5fa; font-weight:700;">📩 Challan #{notif.get('violation_id','?')}</span>
                        <span style="color:#94a3b8; margin-left:10px;">{notif.get('plate','')}</span>
                    </div>
                    <span style="color:#64748b; font-size:0.75rem;">{notif.get('timestamp','')}</span>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:6px;">
                    <div>✉️ Email → <code style="font-size:0.75rem;">{email_ch.get('recipient','—')}</code> <span style="color:{'#22c55e' if 'SENT' in str(email_ch.get('status','')) else '#f59e0b'};">[{email_ch.get('status','—')}]</span></div>
                    <div>📱 SMS → <code style="font-size:0.75rem;">{sms_ch.get('recipient','—')}</code> <span style="color:{'#22c55e' if 'DELIVERED' in str(sms_ch.get('status','')) or 'SENT' in str(sms_ch.get('status','')) else '#f59e0b'};">[{sms_ch.get('status','—')}]</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        render_empty_state("No Notifications Dispatched Yet", "Use the 'Dispatch SMS / Email Notice' button in the Violations Database tab or pay a challan to trigger automated notices.", "📭")

# ---------------------------------------------------------
# TAB 4: OCR Preprocessing Lab
# ---------------------------------------------------------
with tab_ocr_playground:
    render_section_header(
        "OCR Preprocessing Lab",
        "Visualize how image crops are enhanced to boost plate recognition accuracy",
        "🔬",
    )

    col_play_input, col_play_results = st.columns([1, 2])

    with col_play_input:
        render_panel_start("panel-input")
        render_section_header("Crop Selection", "Pick a sample plate or upload your own", "🖼️")
        # List synthetic files
        sample_plates_dir = "test_assets"
        os.makedirs(sample_plates_dir, exist_ok=True)
        plate_files = [f for f in os.listdir(sample_plates_dir) if "plate" in f and f.endswith(('.png', '.jpg', '.jpeg'))]
        
        selected_plate = None
        if plate_files:
            selected_plate = st.selectbox("Select Cropped Plate", plate_files)
            plate_img_path = os.path.join(sample_plates_dir, selected_plate)
            raw_plate_img = cv2.imread(plate_img_path)
        else:
            raw_plate_img = None
            render_empty_state("No plate samples", "Generate demo assets from the Detector tab", "🔤")

        custom_uploaded_plate = st.file_uploader("Or Upload Custom Bounding Box Plate Crop", type=["jpg", "png", "jpeg"])
        if custom_uploaded_plate is not None:
            file_bytes = np.asarray(bytearray(custom_uploaded_plate.read()), dtype=np.uint8)
            raw_plate_img = cv2.imdecode(file_bytes, 1)
        render_panel_end()

    with col_play_results:
        render_panel_start("panel-output")
        if raw_plate_img is not None:
            from backend.image_enhancer import enhance as enhance_image, get_luminance
            
            # Night-Vision Low-Light Enhancement
            enh_res = enhance_image(raw_plate_img, force=force_enhancer)
            enhanced_bgr = enh_res["enhanced"]
            lum_in = enh_res["luminance_before"]
            lum_out = enh_res["luminance_after"]
            stages_txt = ", ".join(enh_res["stages_applied"]) if enh_res["stages_applied"] else "None (Well-lit)"

            st.markdown(f"#### 🌙 Night-Vision & Preprocessing Pipeline")
            st.caption(f"**Luminance Before:** {lum_in:.1f} | **After Boost:** {lum_out:.1f} | **Stages Applied:** `{stages_txt}`")

            # Preprocessing on enhanced image
            gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
            factor = ocr_engine.preprocess_settings.get("resize_factor", 2.0)
            h, w = gray.shape[:2]
            resized = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
            filtered = cv2.bilateralFilter(resized, 11, 17, 17)
            deskewed = ocr_engine.deskew(filtered)
            _, thresholded = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edge_pixels = np.concatenate([thresholded[0, :], thresholded[-1, :], thresholded[:, 0], thresholded[:, -1]])
            if np.mean(edge_pixels) < 127:
                thresholded = cv2.bitwise_not(thresholded)
                
            # Render processing columns
            c_orig, c_enh, c_gray, c_filtered, c_bin = st.columns(5)
            with c_orig:
                st.image(cv2.cvtColor(raw_plate_img, cv2.COLOR_BGR2RGB), caption=f"1. Raw Crop ({lum_in:.0f} L)", use_container_width=True)
            with c_enh:
                st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption=f"2. Night Boost ({lum_out:.0f} L)", use_container_width=True)
            with c_gray:
                st.image(gray, caption="3. Grayscale 2x", use_container_width=True)
            with c_filtered:
                st.image(filtered, caption="4. Bilateral Filter", use_container_width=True)
            with c_bin:
                st.image(thresholded, caption="5. Otsu Binary", use_container_width=True)
                
            st.divider()
            
            st.write("#### OCR Execution Comparison")
            # We run both Tesseract and EasyOCR on original vs preprocessed vs night-enhanced to demonstrate boost
            t_orig_text, t_orig_conf = ocr_engine.run_tesseract(cv2.cvtColor(raw_plate_img, cv2.COLOR_BGR2GRAY))
            t_proc_text, t_proc_conf = ocr_engine.run_tesseract(thresholded)
            
            e_orig_text, e_orig_conf = ocr_engine.run_easyocr(raw_plate_img)
            e_proc_text, e_proc_conf = ocr_engine.run_easyocr(thresholded)
            
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.markdown("### Google Tesseract OCR")
                st.markdown(f"""
                - **Raw Image Output:** `{ocr_engine.clean_plate_text(t_orig_text) or "Failed"}` *(Confidence: {t_orig_conf*100:.1f}%)*
                - **Preprocessed Output:** <span class="metric-value">{ocr_engine.clean_plate_text(t_proc_text) or "Failed"}</span> *(Confidence: {t_proc_conf*100:.1f}%)*
                """, unsafe_allow_html=True)
                
            with res_c2:
                st.markdown("### JaidedAI EasyOCR")
                st.markdown(f"""
                - **Raw Image Output:** `{ocr_engine.clean_plate_text(e_orig_text) or "Failed"}` *(Confidence: {e_orig_conf*100:.1f}%)*
                - **Preprocessed Output:** <span class="metric-value">{ocr_engine.clean_plate_text(e_proc_text) or "Failed"}</span> *(Confidence: {e_proc_conf*100:.1f}%)*
                """, unsafe_allow_html=True)
        else:
            render_empty_state(
                "No plate selected",
                "Choose a sample or upload a cropped plate image",
                "🔤",
            )
        render_panel_end()

# ---------------------------------------------------------
# TAB 4: RTO Vehicle Registry
# ---------------------------------------------------------
with tab_rto:
    render_section_header(
        "RTO Vehicle Registry",
        "Query the motor vehicle registration database",
        "🔍",
    )

    quick_plates = ["MH12DE5678", "MH12AB1234", "DL3CAY1111", "KA03MG9999", "HR26BP0007"]

    if "rto_query_input" not in st.session_state:
        st.session_state.rto_query_input = ""

    render_quick_plate_chips(quick_plates, key_prefix="btn_rto")

    render_panel_start()
    with st.form("rto_search_form"):
        search_plate = st.text_input(
            "Enter License Plate Number (e.g. MH12AB1234, DL4C1234)",
            value=st.session_state.rto_query_input,
            key="rto_text_input",
            placeholder="MH12AB1234",
        )
        submit_search = st.form_submit_button("🔎 Query Vehicle Details", use_container_width=True)
    render_panel_end()

    if submit_search or (search_plate and not submit_search):
        if not search_plate.strip():
            st.warning("Please enter a valid license plate number.")
        else:
            with st.spinner("Querying RTO database / API..."):
                time.sleep(0.3)  # Small realistic latency simulation
                rto_info = query_rto(search_plate, config=config)
                
                # Check status and set badge classes
                is_insured = "Active" in rto_info.get("insurance_status", "")
                ins_badge = "rc-badge-active" if is_insured else "rc-badge-expired"
                ins_label = rto_info.get("insurance_status", "Expired")
                
                status_badge = "rc-badge-active" if rto_info.get("status") == "Active" else "rc-badge-expired"
                status_label = rto_info.get("status", "Active")
                
                clean_plate_no = re.sub(r'[^A-Z0-9]', '', search_plate.upper())
                
                # Render the premium smart RC card using HTML
                st.markdown(f"""
                <div class="rc-card">
                    <div class="rc-header">
                        <span>MINISTRY OF ROAD TRANSPORT & HIGHWAYS</span>
                        <span class="rc-badge {status_badge}">{status_label}</span>
                    </div>
                    <div style="text-align: center; color: #94a3b8; font-size: 0.65rem; text-transform: uppercase; margin-bottom: 5px;">
                        GOVERNMENT OF INDIA | REGISTRATION CERTIFICATE
                    </div>
                    <div class="rc-plate-container">
                        <div class="rc-plate">{clean_plate_no}</div>
                    </div>
                    <div class="rc-grid">
                        <div class="rc-field">
                            <div class="rc-label">Registered Owner</div>
                            <div class="rc-value">{rto_info.get('owner_name', 'N/A')}</div>
                        </div>
                        <div class="rc-field">
                            <div class="rc-label">Maker / Model</div>
                            <div class="rc-value">{rto_info.get('vehicle_model', 'N/A')}</div>
                        </div>
                        <div class="rc-field">
                            <div class="rc-label">RTO District Office</div>
                            <div class="rc-value" style="color: #f59e0b; font-weight: 600;">{rto_info.get('rto_office', 'N/A')}</div>
                        </div>
                        <div class="rc-field">
                            <div class="rc-label">Fuel Type</div>
                            <div class="rc-value">{rto_info.get('fuel_type', 'N/A')}</div>
                        </div>
                        <div class="rc-field">
                            <div class="rc-label">Registration Date</div>
                            <div class="rc-value">{rto_info.get('registration_date', 'N/A')}</div>
                        </div>
                        <div class="rc-field">
                            <div class="rc-label">Insurance Status</div>
                            <div>
                                <span class="rc-badge {ins_badge}">{ins_label}</span>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; font-size: 0.65rem; color: #64748b; text-align: center;">
                        Database Source: {rto_info.get('api_source', 'RTO Parivahan Vahan Registry')} | Chip Serial: MoRTH-CS-{abs(hash(clean_plate_no)) % 100000000:08d}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5: Analytics Dashboard
# ---------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=["#f59e0b", "#ef4444", "#22c55e", "#818cf8", "#fb7185", "#a78bfa"],
)

with tab_analytics:
    render_section_header(
        "Analytics Dashboard",
        "Trends, patterns, and insights derived from all recorded violations",
        "📈",
    )

    col_reload_a, col_spacer_a = st.columns([1, 4])
    with col_reload_a:
        if st.button("🔄 Refresh Analytics", type="secondary", use_container_width=True, key="btn_analytics_reload"):
            st.cache_resource.clear()
            st.rerun()

    analytics = db.get_analytics_data()

    if analytics["total"] == 0:
        render_empty_state(
            "No data to analyse",
            "Run the detector and log some violations to see charts appear here.",
            "📊",
        )
    else:
        # ── KPI Row ─────────────────────────────────────────────────────────
        total = analytics["total"]
        total_fines = analytics["total_fines"]
        daily = analytics["daily_series"]
        avg_daily = round(total / max(len(daily), 1), 1)

        render_metric_cards([
            {"icon": "🚨", "label": "Total Violations", "value": total},
            {"icon": "💰", "label": "Total Fines Collected", "value": f"₹{total_fines:,.0f}"},
            {"icon": "📅", "label": "Avg Violations / Day", "value": avg_daily},
            {"icon": "🗺️", "label": "States Represented",
             "value": len(analytics["state_counts"]) if not analytics["state_counts"].empty else 0},
        ])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GIS & 2D Surveillance Hotspot Heatmap ─────────────────────────
        st.markdown("### 🗺️ GIS Traffic Violation & Camera Hotspot Density")
        col_map1, col_map2 = st.columns([3, 2], gap="medium")
        
        with col_map1:
            hotspot_data = pd.DataFrame([
                {"City": "Mumbai (Western Express Hwy)", "lat": 19.0760, "lon": 72.8777, "Violations": 142},
                {"City": "Delhi (Ring Road Crossing)", "lat": 28.6139, "lon": 77.2090, "Violations": 198},
                {"City": "Bengaluru (Outer Ring Rd)", "lat": 12.9716, "lon": 77.5946, "Violations": 165},
                {"City": "Pune (MG Road Junction)", "lat": 18.5204, "lon": 73.8567, "Violations": 115},
                {"City": "Hyderabad (HITEC Corridor)", "lat": 17.3850, "lon": 78.4867, "Violations": 94},
                {"City": "Chennai (Anna Salai)", "lat": 13.0827, "lon": 80.2707, "Violations": 88},
            ])
            
            fig_map = px.scatter_geo(
                hotspot_data,
                lat="lat", lon="lon",
                hover_name="City",
                size="Violations",
                color="Violations",
                color_continuous_scale="Oranges",
                projection="natural earth",
                title="National Traffic Violation Hotspot Density",
            )
            fig_map.update_geos(
                scope="asia",
                center=dict(lat=20.5937, lon=78.9629),
                projection_scale=4.5,
                showland=True, landcolor="rgba(15,23,42,0.9)",
                showocean=True, oceancolor="rgba(10,15,30,0.95)",
                showcountries=True, countrycolor="rgba(245,158,11,0.2)",
                showsubunits=True, subunitcolor="rgba(245,158,11,0.1)",
                bgcolor="rgba(0,0,0,0)"
            )
            fig_map.update_layout(**_PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig_map, use_container_width=True)

        with col_map2:
            st.markdown("#### 📹 Camera Surveillance Hotspots")
            cam_df = analytics.get("camera_hotspots", pd.DataFrame())
            if not cam_df.empty:
                fig_cam = px.bar(
                    cam_df,
                    x="Violations",
                    y="Location",
                    orientation="h",
                    color="Violations",
                    color_continuous_scale="Reds",
                    title="CCTV Sectors by Flagged Incidents",
                )
                fig_cam.update_layout(**_PLOTLY_LAYOUT, height=360, yaxis=dict(autorange="reversed", showgrid=False))
                st.plotly_chart(fig_cam, use_container_width=True)
            else:
                st.caption("No camera sector data recorded yet.")

        st.divider()

        # ── 2D Day-of-Week × Time-of-Day Risk Intensity Matrix Heatmap ───────
        st.markdown("### 🔥 24/7 Traffic Risk Intensity Matrix (Day × Hour Heatmap)")
        matrix_df = analytics.get("weekday_hour_matrix", pd.DataFrame())
        if not matrix_df.empty and (matrix_df.values.sum() > 0 or total > 0):
            fig_matrix = px.imshow(
                matrix_df,
                labels=dict(x="Hour of Day", y="Day of Week", color="Violations"),
                x=list(matrix_df.columns),
                y=list(matrix_df.index),
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title="Violation High-Risk Enforcement Hot-Hours Matrix"
            )
            fig_matrix.update_layout(
                **_PLOTLY_LAYOUT,
                height=320,
                xaxis=dict(tickangle=-45, showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
        else:
            st.caption("Not enough timestamp distribution to render 2D risk matrix.")

        st.divider()

        # ── Multi-Violation Type & Speed Distribution Row ──────────────────
        col_vtype, col_speed = st.columns(2, gap="large")

        with col_vtype:
            st.markdown("### 🚨 Violation Breakdown (Multi-Category)")
            vt_df = analytics.get("violation_types", pd.DataFrame())
            if not vt_df.empty:
                fig_vt = px.pie(
                    vt_df,
                    names="Violation Type",
                    values="Count",
                    hole=0.45,
                    color_discrete_sequence=["#ef4444", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"],
                    title="Proportion of Offense Categories"
                )
                fig_vt.update_traces(textinfo="label+percent", textfont=dict(size=12))
                fig_vt.update_layout(**_PLOTLY_LAYOUT, showlegend=True)
                st.plotly_chart(fig_vt, use_container_width=True)
            else:
                st.caption("No violation type records available.")

        with col_speed:
            st.markdown("### ⚡ Recorded Vehicle Speed Distribution")
            spd_df = analytics.get("speed_dist", pd.DataFrame())
            if not spd_df.empty:
                fig_spd = px.bar(
                    spd_df,
                    x="Speed Range",
                    y="Vehicle Count",
                    color="Vehicle Count",
                    color_continuous_scale="Teal",
                    title="Radar Speed Measurements (km/h)"
                )
                fig_spd.update_layout(**_PLOTLY_LAYOUT, xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
                st.plotly_chart(fig_spd, use_container_width=True)
            else:
                st.caption("Speed telemetry will populate as vehicles are tracked in video / CCTV streams.")

        st.divider()

        # ── Row 1: Daily trend area chart ───────────────────────────────────
        st.markdown("### 📅 Violations Over Time")
        if not daily.empty:
            fig_daily = px.area(
                daily, x="Date", y="Violations",
                title="Daily Violation Count",
                labels={"Violations": "Violations", "Date": ""},
                color_discrete_sequence=["#f59e0b"],
            )
            fig_daily.update_traces(
                line=dict(width=2.5),
                fillcolor="rgba(245,158,11,0.12)",
            )
            fig_daily.update_layout(**_PLOTLY_LAYOUT)
            fig_daily.update_xaxes(showgrid=False)
            fig_daily.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.caption("Not enough timestamped data.")

        st.divider()

        # ── Row 2: Hourly bar chart + State donut pie ───────────────────────
        col_hourly, col_state = st.columns(2, gap="large")

        with col_hourly:
            st.markdown("### ⏰ Violations by Hour of Day")
            hourly_df = analytics["hourly_series"]
            if not hourly_df.empty:
                peak_hour = int(hourly_df.loc[hourly_df["Violations"].idxmax(), "Hour"])
                colors = [
                    "#ef4444" if h == peak_hour else "#f59e0b"
                    for h in hourly_df["Hour"]
                ]
                fig_hr = go.Figure(
                    go.Bar(
                        x=hourly_df["Hour"],
                        y=hourly_df["Violations"],
                        marker_color=colors,
                        name="Violations",
                    )
                )
                fig_hr.update_layout(
                    **_PLOTLY_LAYOUT,
                    title=f"Peak Hour: {peak_hour:02d}:00",
                    xaxis=dict(title="Hour (24h)", dtick=2, showgrid=False),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                )
                st.plotly_chart(fig_hr, use_container_width=True)
            else:
                st.caption("No hourly data available.")

        with col_state:
            st.markdown("### 🗺️ Violations by Registered State")
            state_df = analytics["state_counts"]
            if not state_df.empty:
                fig_state = px.pie(
                    state_df,
                    names="State",
                    values="Violations",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                )
                fig_state.update_traces(
                    textinfo="label+percent",
                    textfont=dict(size=12),
                )
                fig_state.update_layout(**_PLOTLY_LAYOUT, showlegend=True)
                st.plotly_chart(fig_state, use_container_width=True)
            else:
                st.caption("Not enough state data to visualize.")

        st.divider()

        # ── Row 3: Manufacturer horizontal bar + OCR confidence histogram ───
        col_mfr, col_conf = st.columns(2, gap="large")

        with col_mfr:
            st.markdown("### 🏍️ Violations by Vehicle Manufacturer")
            mfr_df = analytics["manufacturer_counts"]
            if not mfr_df.empty:
                fig_mfr = px.bar(
                    mfr_df.head(10),
                    x="Violations",
                    y="Manufacturer",
                    orientation="h",
                    color="Violations",
                    color_continuous_scale="Viridis",
                    title="Top 10 Manufacturers",
                )
                fig_mfr.update_layout(
                    **_PLOTLY_LAYOUT,
                    yaxis=dict(autorange="reversed", showgrid=False),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_mfr, use_container_width=True)
            else:
                st.caption("Not enough manufacturer data.")

        with col_conf:
            st.markdown("### 🎯 OCR Confidence Distribution")
            conf_df = analytics["confidence_dist"]
            if not conf_df.empty:
                fig_conf = px.bar(
                    conf_df,
                    x="Confidence Band",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Teal",
                    title="Plate-recognition Confidence Bands",
                )
                fig_conf.update_layout(
                    **_PLOTLY_LAYOUT,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.caption("No OCR confidence data recorded yet.")

        st.divider()

        # ── Row 4: Top repeat-offender plates leaderboard ──────────────────
        st.markdown("### 🏆 Top Repeat-Offender Plates")
        top_df = analytics["top_plates"]
        if not top_df.empty:
            fig_top = px.bar(
                top_df,
                x="Occurrences",
                y="Plate",
                orientation="h",
                color="Occurrences",
                color_continuous_scale="Reds",
                title="Most-Seen Plates Across All Sessions",
            )
            fig_top.update_layout(
                **_PLOTLY_LAYOUT,
                yaxis=dict(autorange="reversed", showgrid=False),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", dtick=1),
                coloraxis_showscale=True,
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.caption("No repeat plates recorded.")

        st.divider()

        # ── Export ──────────────────────────────────────────────────────────
        st.markdown("### 📥 Export Analytics Data")
        _ana_df = db.get_all_violations()
        if not _ana_df.empty:
            csv_bytes = _ana_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Violations CSV",
                data=csv_bytes,
                file_name="analytics_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------
# TAB 7: REST API & Edge Hub
# ---------------------------------------------------------
with tab_rest_api:
    render_section_header(
        "REST API & CCTV Edge Node Integration",
        "Connect external CCTV cameras, municipal surveillance streams, and edge processors via FastAPI endpoints.",
        "🔌"
    )
    
    col_api1, col_api2 = st.columns([1, 1], gap="large")
    
    with col_api1:
        st.markdown("### 📡 API Server Status")
        st.info("⚡ **FastAPI Service Endpoint**: `http://localhost:8000`\n\n📖 **Interactive OpenAPI Specs**: [http://localhost:8000/docs](http://localhost:8000/docs)")
        
        st.markdown("#### 🚀 Available Endpoints")
        st.markdown("""
        - `GET /` — Health check & system telemetry
        - `POST /api/v1/scan/image` — Ingest camera frame & generate E-Challan
        - `GET /api/v1/violations` — Query paginated violation records
        - `GET /api/v1/violations/{id}` — Fetch single violation detail
        - `PATCH /api/v1/violations/{id}/status` — Update ticket status (`PENDING`, `PAID`, `CANCELLED`)
        - `POST /api/v1/challan/{id}/pay` — Process UPI / Online E-Challan payment
        - `GET /api/v1/challan/{id}/receipt` — Download official Tax Receipt
        - `GET /api/v1/analytics/stats` — Aggregate metrics & peak hours
        - `GET /api/v1/challan/file/{filename}` — Serve generated E-Challan PNG
        """)
        
    with col_api2:
        st.markdown("### 🐍 Python Client Example")
        st.code("""import requests

url = "http://localhost:8000/api/v1/scan/image"
files = {"file": open("traffic_frame.jpg", "rb")}
params = {"enhance_low_light": True, "auto_generate_challan": True}

response = requests.post(url, files=files, params=params)
print(response.json())
""", language="python")

        st.markdown("### 🐳 Docker & Docker-Compose Deployment")
        st.code("""# Launch full system (Streamlit Dashboard + FastAPI REST API)
docker-compose up -d --build

# View container logs
docker-compose logs -f
""", language="bash")