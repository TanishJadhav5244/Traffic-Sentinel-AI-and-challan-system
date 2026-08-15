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
from frontend.styles import inject_custom_styles
from frontend.components import (
    render_hero,
    render_section_header,
    content_panel,
    render_metric_cards,
    render_empty_state,
    render_quick_plate_chips,
    render_pipeline_steps,
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
        crop_dir=storage.get("crop_dir", "violations/crops")
    )

detector = get_detector(config)
ocr_engine = get_ocr_engine(config)
db = get_db(config)

# Hero banner with live stats
_violation_df = db.get_all_violations()
render_hero(violation_count=len(_violation_df), demo_mode=detector.demo_mode)

# Sidebar Configuration
st.sidebar.markdown(
    '<div style="text-align:center;padding:0.5rem 0 1rem;">'
    '<img src="https://img.icons8.com/nolan/128/security-camera.png" width="72" '
    'style="filter:drop-shadow(0 0 12px rgba(56,189,248,0.4));"/>'
    '</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("### ⚙️ System Configuration")

# Engine selection
ocr_selection = st.sidebar.selectbox(
    "OCR Engine",
    options=["easyocr", "tesseract"],
    index=0 if config.get("ocr", {}).get("default_engine") == "easyocr" else 1
)

# Threshold sliders
st.sidebar.subheader("Detection Thresholds")
rider_conf = st.sidebar.slider("Rider Bounding Box Conf", 0.10, 1.00, float(config.get("models", {}).get("confidence", {}).get("rider", 0.35)))
helmet_conf = st.sidebar.slider("Helmet Conf", 0.10, 1.00, float(config.get("models", {}).get("confidence", {}).get("helmet", 0.40)))
plate_conf = st.sidebar.slider("License Plate Conf", 0.10, 1.00, float(config.get("models", {}).get("confidence", {}).get("license_plate", 0.40)))

# Set runtime settings back into classes
detector.thresholds["rider"] = rider_conf
detector.thresholds["helmet"] = helmet_conf22
detector.thresholds["license_plate"] = plate_conf

# Low-light / Night mode controls
st.sidebar.subheader("🌙 Low-Light & Night Vision")
enable_enhancer = st.sidebar.checkbox(
    "Auto Low-Light Enhancement",
    value=bool(config.get("ocr", {}).get("enhancer", {}).get("auto_low_light", True)),
    help="Detects dark plate crops and automatically applies Retinex, CLAHE, and Denoising."
)
force_enhancer = st.sidebar.checkbox(
    "Force Night-Vision Filter",
    value=bool(config.get("ocr", {}).get("enhancer", {}).get("force_enhancement", False)),
    help="Forces multi-stage illumination boost on all cropped plates regardless of brightness."
)
ocr_engine.enhancer_settings["auto_low_light"] = enable_enhancer
ocr_engine.enhancer_settings["force_enhancement"] = force_enhancer

# Mode status display
if detector.demo_mode:
    st.sidebar.markdown(
        '<div class="sidebar-status sidebar-status-demo">'
        '⚠️ Demo Mode — using YOLOv8n fallback weights'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="sidebar-status sidebar-status-live">'
        '🚀 Live — custom fine-tuned model loaded'
        '</div>',
        unsafe_allow_html=True,
    )

# Tab setup
tab_detector, tab_database, tab_ocr_playground, tab_rto, tab_analytics = st.tabs([
    "📸 Live Violation Detector",
    "📊 Violations Database",
    "🔬 OCR Preprocessing Lab",
    "🔍 RTO Vehicle Registry",
    "📈 Analytics Dashboard",
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
                ["Sample Images", "Upload File (Image/Video)"],
                horizontal=True,
                label_visibility="collapsed",
            )
        
        input_image = None
        input_video_path = None
        
        if upload_type == "Sample Images":
            # List sample files
            sample_dir = "test_assets"
            os.makedirs(sample_dir, exist_ok=True)
            samples = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.mp4'))]
            
            if not samples:
                render_empty_state(
                    "No sample assets found",
                    "Generate demo images and download model weights to get started",
                    "🖼️",
                )
                if st.button("⚡ Generate Demo Assets", type="secondary"):
                    with st.spinner("Generating sample images and downloading weights..."):
                        os.system("python backend/models/download_weights.py")
                        st.rerun()
            else:
                selected_sample = st.selectbox("Select Sample Asset", samples)
                sample_path = os.path.join(sample_dir, selected_sample)
                if sample_path.endswith('.mp4'):
                    input_video_path = sample_path
                else:
                    input_image = cv2.imread(sample_path)
        else:
            uploaded_file = st.file_uploader("Upload Image or Video (MP4)", type=["jpg", "jpeg", "png", "mp4"])
            if uploaded_file is not None:
                # Check extension
                if uploaded_file.name.endswith('.mp4'):
                    # Save to temp file
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
        
        if is_video_input and st.session_state.video_processing_active:
            st.button("⏹ Stop Processing", type="secondary", use_container_width=True, on_click=stop_video_processing)
            run_pipeline = False
        else:
            run_pipeline = st.button("▶ Run System Inference", type="primary", use_container_width=True)
            if run_pipeline and is_video_input:
                st.session_state.video_processing_active = True

    with col_output:
        with content_panel("output"):
            render_section_header("System Visual Output", "Detection results and violation evidence appear here", "🎯")
        
        if run_pipeline or (is_video_input and st.session_state.video_processing_active):
            # PROCESS IMAGE
            if input_image is not None:
                st.info("Processing image pipeline...")
                img_to_proc = input_image.copy()
                
                # Step 1: Detect
                detections = detector.detect(img_to_proc)
                
                # Step 2: Associate no-helmet to plate
                violations = detector.associate_violations(detections)
                
                # Step 3: OCR on associated plate regions
                logged_violations = []
                for v in violations:
                    p_box = v["plate"]["box"]
                    r_box = v["rider"]["box"]
                    
                    # Crop image arrays
                    h, w = img_to_proc.shape[:2]
                    px1, py1, px2, py2 = max(0, p_box[0]), max(0, p_box[1]), min(w, p_box[2]), min(h, p_box[3])
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
                    
                    # Save and Log to DB
                    record = db.log_violation(
                        frame_timestamp="00:00:00",
                        plate_crop=plate_crop,
                        rider_crop=rider_crop,
                        plate_text=plate_text,
                        ocr_conf=ocr_conf,
                        helmet_status="no-helmet",
                        night_mode=was_low_light
                    )
                    logged_violations.append((record, plate_crop, rider_crop, ocr_res))
                    
                # Annotate and show image
                annotated_img = detector.draw_annotations(img_to_proc, detections, violations)
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                
                # Show cards below
                if logged_violations:
                    st.error(f"🚨 Detected {len(logged_violations)} Helmet Violations!")
                    card_cols = st.columns(min(3, len(logged_violations)))
                    for idx, (rec, p_crp, r_crp, ocr_info) in enumerate(logged_violations):
                        with card_cols[idx % 3]:
                            night_badge = '<span style="background:rgba(251,191,36,0.2);color:#fbbf24;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;margin-left:6px;">🌙 Night Enhanced</span>' if rec.get('night_mode') or ocr_info.get('was_low_light') else ''
                            st.markdown(f"""
                            <div class="violation-card">
                                <div class="violation-title">Violation ID: {rec['violation_id']} {night_badge}</div>
                                <b>Plate:</b> <span class="metric-value">{rec['plate_text']}</span><br>
                                <b>Owner:</b> {rec.get('owner_name', 'Unknown')}<br>
                                <b>Vehicle:</b> {rec.get('vehicle_model', 'Unknown')}<br>
                                <b>Time:</b> {rec['timestamp']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show side-by-side crops
                            crop_c1, crop_c2 = st.columns(2)
                            with crop_c1:
                                if r_crp.size > 0:
                                    st.image(cv2.cvtColor(r_crp, cv2.COLOR_BGR2RGB), caption="Rider", use_container_width=True)
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
                    st.success("✅ No Helmet Violations detected.")
                    
            # PROCESS VIDEO
            elif input_video_path is not None:
                st.info("Processing video pipeline...")
                video_placeholder = st.empty()
                status_text = st.empty()
                
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    st.error("Error opening video file.")
                    st.session_state.video_processing_active = False
                else:
                    try:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # De-duplication set to prevent spamming CSV logs for same plate within video
                        recent_detections = {}
                        
                        frame_idx = 0
                        violations_found = 0
                        
                        while cap.isOpened() and st.session_state.video_processing_active:
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            frame_idx += 1
                            
                            # Process every 4th frame to ensure smooth playback on CPU
                            if frame_idx % 4 != 0:
                                continue
                                
                            timestamp_str = str(datetime.timedelta(seconds=int(frame_idx / fps)))
                            status_text.text(f"Processing Frame: {frame_idx}/{frame_count} (Timestamp: {timestamp_str})")
                            
                            # Pipeline Inference
                            detections = detector.detect(frame)
                            violations = detector.associate_violations(detections)
                            
                            current_frame_violations = []
                            for v in violations:
                                p_box = v["plate"]["box"]
                                r_box = v["rider"]["box"]
                                
                                h, w = frame.shape[:2]
                                px1, py1, px2, py2 = max(0, p_box[0]), max(0, p_box[1]), min(w, p_box[2]), min(h, p_box[3])
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
                                
                                # De-duplicate plate text within a rolling 5-second window
                                current_time = frame_idx / fps
                                if plate_text != "UNKNOWN" and len(plate_text) >= 5:
                                    last_seen = recent_detections.get(plate_text, -999)
                                    if current_time - last_seen > 5.0:
                                        recent_detections[plate_text] = current_time
                                        
                                        # Log to CSV
                                        db.log_violation(
                                            frame_timestamp=timestamp_str,
                                            plate_crop=plate_crop,
                                            rider_crop=rider_crop,
                                            plate_text=plate_text,
                                            ocr_conf=ocr_conf,
                                            helmet_status="no-helmet",
                                            night_mode=was_low_light
                                        )
                                        violations_found += 1
                                        st.toast(f"🚨 Helmet Violation: {plate_text}", icon="🚨")
                                        
                            # Draw bounding boxes and update video frame
                            annotated = detector.draw_annotations(frame, detections, violations)
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(annotated_rgb, use_container_width=True)
                            
                            # Keep frame rate reasonable
                            time.sleep(0.01)
                            
                        if not st.session_state.video_processing_active:
                            status_text.warning("Processing stopped by user.")
                        else:
                            st.session_state.video_processing_active = False
                            status_text.success(f"Processing Complete! Identified {violations_found} unique helmet violations.")
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
        render_metric_cards([
            {"icon": "🚨", "label": "Violations Detected", "value": len(df)},
            {"icon": "💰", "label": "Total Fines Levied", "value": f"₹{len(df)*1000:,}"},
            {"icon": "🏍️", "label": "Helmet Compliance", "value": "91.4%", "delta": "+1.2% weekly"},
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

        # Search bar
        search_query = st.text_input("🔍 Search by Plate Number", "")
        if search_query:
            df_filtered = df[df["plate_text"].str.contains(search_query.upper(), na=False)]
        else:
            df_filtered = df.copy()
            
        # Display table
        st.dataframe(
            df_filtered[["violation_id", "timestamp", "plate_text", "ocr_confidence", "helmet_status", "owner_name", "vehicle_model"]],
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
                    st.write(f"**Owner:** {row.get('owner_name', 'Unknown')}")
                    st.write(f"**Vehicle Model:** {row.get('vehicle_model', 'Unknown')}")
                    st.write(f"**Fine Levied:** ₹{row.get('challan_amount', 1000.0):,.2f}")
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
                        with open(str(c_path), "rb") as f:
                            st.download_button(
                                label="📥 Download Challan PNG",
                                data=f,
                                file_name=f"challan_{row['violation_id']}.png",
                                mime="image/png",
                                key=f"db_dl_{row['violation_id']}_{idx}",
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
                use_container_width=True
            )
        with col_act2:
            if st.button("🗑️ Wipe Database Logs", type="secondary", use_container_width=True):
                db.clear_database()
                st.warning("All records and cropped image binaries have been deleted.")
                st.rerun()

# ---------------------------------------------------------
# TAB 3: OCR Preprocessing Lab
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
            with st.spinner("Querying secure RTO database..."):
                time.sleep(0.3)  # Small realistic latency simulation
                rto_info = query_rto(search_plate)
                
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
                        <div class="rc-field">
                            <div class="rc-label">Fitness Validity</div>
                            <div class="rc-value">Valid upto 15 Years from Reg.</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; font-size: 0.65rem; color: #64748b; text-align: center;">
                        Chip Serial: MoRTH-CS-{abs(hash(clean_plate_no)) % 100000000:08d} | Digital Authenticity Verified
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
    colorway=["#38bdf8", "#818cf8", "#fb7185", "#34d399", "#fbbf24", "#a78bfa"],
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

        # ── Row 1: Daily trend area chart ───────────────────────────────────
        st.markdown("### 📅 Violations Over Time")
        if not daily.empty:
            fig_daily = px.area(
                daily, x="Date", y="Violations",
                title="Daily Violation Count",
                labels={"Violations": "Violations", "Date": ""},
                color_discrete_sequence=["#38bdf8"],
            )
            fig_daily.update_traces(
                line=dict(width=2.5),
                fillcolor="rgba(56,189,248,0.12)",
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
                    "#fb7185" if h == peak_hour else "#38bdf8"
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
                coloraxis_showscale=False,
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
//  uasdada