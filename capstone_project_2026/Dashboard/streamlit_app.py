"""Streamlit Cloud entry point for the static Data Roles dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parent
HTML_PATH = DASHBOARD_DIR / "index.html"
DATA_SCRIPT_PATH = DASHBOARD_DIR / "jobs.js"
DATA_SCRIPT_TAG = '<script src="./jobs.js"></script>'


st.set_page_config(
    page_title="Signal / Data Roles Index",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the job index…")
def load_dashboard_html() -> str:
    """Inline the generated data so the HTML works inside Streamlit's iframe."""

    if not HTML_PATH.exists():
        raise FileNotFoundError(f"Dashboard HTML not found: {HTML_PATH}")
    if not DATA_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Dashboard data not found: {DATA_SCRIPT_PATH}")

    html = HTML_PATH.read_text(encoding="utf-8")
    data_script = DATA_SCRIPT_PATH.read_text(encoding="utf-8")
    if DATA_SCRIPT_TAG not in html:
        raise RuntimeError(
            "Dashboard HTML is missing the generated jobs.js script tag."
        )

    # Prevent a description containing </script> from closing the inline tag.
    safe_data_script = data_script.replace("</script", "<\\/script")
    return html.replace(
        DATA_SCRIPT_TAG,
        f"<script>{safe_data_script}</script>",
        1,
    )


try:
    dashboard_html = load_dashboard_html()
except (FileNotFoundError, RuntimeError) as exc:
    st.error("The dashboard could not be loaded.")
    st.code(str(exc))
    st.stop()


st.iframe(
    dashboard_html,
    height=2_400,
)
