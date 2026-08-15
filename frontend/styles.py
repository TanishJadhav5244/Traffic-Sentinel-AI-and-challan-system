import os

# pyrefly: ignore [missing-import]
import streamlit as st

_CSS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "css", "styles.css")


def inject_custom_styles():
    """Load and inject the project CSS into the Streamlit app."""
    with open(_CSS_PATH, "r", encoding="utf-8") as css_file:
        css_content = css_file.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
