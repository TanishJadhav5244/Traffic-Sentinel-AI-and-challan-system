from contextlib import contextmanager

import streamlit as st


def render_hero(violation_count: int = 0, demo_mode: bool = False):
    mode_label = "Demo Mode" if demo_mode else "Live Model"
    mode_class = "status-demo" if demo_mode else "status-live"
    st.markdown(f"""
    <div class="hero-banner fade-in">
        <div class="hero-glow"></div>
        <div class="hero-content">
            <div class="hero-badge">AI-Powered Traffic Enforcement</div>
            <h1 class="hero-title">Traffic Sentinel AI</h1>
            <p class="hero-subtitle">
                Real-time helmet violation detection, license plate OCR, and automated e-challan generation
            </p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span class="hero-stat-value">{violation_count}</span>
                    <span class="hero-stat-label">Violations Logged</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-value">4</span>
                    <span class="hero-stat-label">Active Modules</span>
                </div>
                <div class="hero-stat">
                    <span class="hero-stat-value">
                        <span class="status-dot {mode_class}"></span>{mode_label}
                    </span>
                    <span class="hero-stat-label">System Status</span>
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
