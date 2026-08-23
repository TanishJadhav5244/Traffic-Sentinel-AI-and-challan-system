from contextlib import contextmanager

# pyrefly: ignore [missing-import]
import streamlit as st


def render_gov_header():
    """Renders top official Government of India / MoRTH portal banner."""
    st.markdown("""
    <div class="gov-top-bar">
        <div class="gov-top-left">
            <span class="gov-emblem">🏛️</span>
            <div class="gov-text-group">
                <span class="gov-authority">GOVERNMENT OF INDIA • MINISTRY OF ROAD TRANSPORT & HIGHWAYS</span>
                <span class="gov-sub-authority">National Intelligent Traffic Enforcement Portal (e-Challan AI System)</span>
            </div>
        </div>
        <div class="gov-top-right">
            <span class="gov-badge-official">OFFICIAL PORTAL</span>
            <span class="gov-helpline">📞 Helpline: 1033 / 112</span>
        </div>
    </div>
    <div class="gov-tricolor-strip"></div>
    """, unsafe_allow_html=True)


def render_hero(violation_count: int = 0):
    mode_label = "Live AI Surveillance Active"
    mode_class = "status-live"
    st.markdown(f"""
    <div class="hero-banner fade-in">
        <div class="hero-glow"></div>
        <div class="hero-content">
            <div class="hero-badge">🏛️ MoRTH Integrated Traffic Intelligence & e-Challan Engine</div>
            <h1 class="hero-title">Traffic Sentinel AI Portal</h1>
            <p class="hero-subtitle">
                Automated helmet violation detection, high-speed license plate OCR, and instant e-challan generation grid
            </p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span class="hero-stat-value">{violation_count}</span>
                    <span class="hero-stat-label">Violations Recorded</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-value">4</span>
                    <span class="hero-stat-label">Active Modules</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-value">
                        <span class="status-dot {mode_class}"></span>{mode_label}
                    </span>
                    <span class="hero-stat-label">Grid Status</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "", icon: str = ""):
    icon_html = f'<span class="section-icon">{icon}</span>' if icon else ""
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        {icon_html}
        <div>
            <h3 class="section-title">{title}</h3>
            {subtitle_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


@contextmanager
def content_panel(variant: str = "default"):
    """Native Streamlit bordered container — widgets stay inside the panel."""
    marker_class = f"panel-marker-{variant}" if variant != "default" else ""
    if marker_class:
        st.markdown(f'<div class="{marker_class}"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        yield


def render_metric_cards(metrics: list[dict]):
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            delta = metric.get("delta", "")
            delta_class = "metric-delta-up" if delta.startswith("+") else "metric-delta-down" if delta.startswith("-") else ""
            delta_html = f'<span class="metric-delta {delta_class}">{delta}</span>' if delta else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{metric.get("icon", "")}</div>
                <div class="metric-label">{metric["label"]}</div>
                <div class="metric-number">{metric["value"]}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)


def render_empty_state(message: str, hint: str = "", icon: str = "📭"):
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">{icon}</div>
        <p class="empty-message">{message}</p>
        <p class="empty-hint">{hint}</p>
    </div>
    """, unsafe_allow_html=True)


def render_quick_plate_chips(plates: list[str], key_prefix: str = "chip", session_key: str = "rto_query_input"):
    st.markdown('<p class="chip-section-label">Quick-Test Plates — click to auto-fill</p>', unsafe_allow_html=True)
    cols = st.columns(len(plates))
    for plate in plates:
        with cols[plates.index(plate)]:
            if st.button(plate, key=f"{key_prefix}_{plate}", use_container_width=True):
                st.session_state[session_key] = plate
                st.session_state["rto_text_input"] = plate
                st.rerun()


def render_pipeline_steps(current_step: int = 0):
    steps = ["Upload", "Detect", "OCR", "Log & Challan"]
    parts = []
    for i, step in enumerate(steps):
        state = "done" if i < current_step else "active" if i == current_step else ""
        parts.append(
            f'<div class="pipeline-step {state}">'
            f'<div class="step-circle">{i + 1}</div>'
            f'<span class="step-label">{step}</span></div>'
        )
        if i < len(steps) - 1:
            parts.append('<div class="step-connector"></div>')
    st.markdown(f'<div class="pipeline-track">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_cctv_hud_header(cam_id: str = "CAM-01", location: str = "MG Road Crossing", fps: int = 30):
    """Renders visual CCTV stream HUD overlay bar."""
    st.markdown(f"""
    <div class="cctv-hud">
        <div class="cctv-hud-left">
            <span class="cctv-rec-dot"></span>
            <span class="cctv-rec-text">REC ● 1080p {fps}FPS</span>
            <span class="cctv-cam-badge">{cam_id}</span>
            <span class="cctv-location">📍 {location}</span>
        </div>
        <div class="cctv-hud-right">
            <span class="cctv-timestamp">AI LIVE SURVEILLANCE FEED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_telemetry_badge(detector_ms: float = 14.5, ocr_ms: float = 38.2, fps: float = 24.5):
    """Renders real-time AI hardware telemetry bar."""
    st.markdown(f"""
    <div class="telemetry-bar">
        <div class="telemetry-item">
            <span class="telemetry-label">YOLOv8 Latency</span>
            <span class="telemetry-val">{detector_ms:.1f} ms</span>
        </div>
        <div class="telemetry-divider"></div>
        <div class="telemetry-item">
            <span class="telemetry-label">OCR Speed</span>
            <span class="telemetry-val">{ocr_ms:.1f} ms</span>
        </div>
        <div class="telemetry-divider"></div>
        <div class="telemetry-item">
            <span class="telemetry-label">Throughput</span>
            <span class="telemetry-val">{fps:.1f} FPS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_chip(status: str) -> str:
    """Returns HTML for styled status badges (Pending, Paid, Disputed)."""
    s = str(status).strip().capitalize()
    if s == "Paid":
        return '<span style="background:#dcfce7; color:#15803d; border:1px solid #86efac; padding:2px 8px; border-radius:12px; font-weight:600; font-size:0.8rem;">🟢 Paid</span>'
    elif s == "Disputed":
        return '<span style="background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; padding:2px 8px; border-radius:12px; font-weight:600; font-size:0.8rem;">🔴 Disputed</span>'
    else:
        return '<span style="background:#fef3c7; color:#b45309; border:1px solid #fcd34d; padding:2px 8px; border-radius:12px; font-weight:600; font-size:0.8rem;">🟡 Pending</span>'

