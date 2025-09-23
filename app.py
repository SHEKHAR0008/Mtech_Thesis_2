# app.py

import streamlit as st
import platform
import psutil

# Import all page functions
from content.main_page import main_page
from content.data_input import data_input_page
from content.data_validation import data_validation_page
from content.adjustment import adjustment_page
from content.visualization import visualization_page
from content.download import download_page

# Import helper functions
from utils.ui_helpers import setup_navigation_buttons

# ======================== Backend Imports and Error Handling ========================
BACKEND_IMPORT_ERROR = None
try:
    from backend.data_io import parse_baseline_text
    from backend.loop_check import check_all_loops
    from backend.observation_equation import build_observation_system
    from backend.initial_guess import initial_guess
    from backend.batch_adjustment import batch_adjustment
    from backend.apply_constraint_fn import apply_constraints
    from backend.csv_result import export_adjustment_results_csv
    from backend.report import generate_adjustment_report_docx_pdf
except Exception as e:
    BACKEND_IMPORT_ERROR = e
if BACKEND_IMPORT_ERROR:
    st.sidebar.error(f"Backend import error: {BACKEND_IMPORT_ERROR}")

# ======================== Session State Initialization ========================
def init_state():
    """Initializes the session state with default values."""
    ss = st.session_state
    defaults = {
        "sidebar_open": False,
        "current_step": "data input",
        "steps_done": {
            "data input": False,
            "Data validation": False,
            "adjustment": False,
            "visualization": False,
            "Download": False,
        },
        "dimension":None,
        "viewed_results": False, # New variable to track if results have been viewed
        "final_results" : None,
        "vtpv_graph" : None,
        "chi_graph": None,
        "network_plot": None,
        "error_ellipse_plot": None,
        "baseline_text": "",
        "unq_stations": None,
        "constrained_text": "",
        "adjustment_type": "Batch Adjustment",
        "weight_matrix": "Unity",
        "sys_info": None,
        "baseline_list": None,
        "loop_ok": None,
        "loop_msg": "",
        "station_list": [],
        "known_stations": [],
        "blocks": None,
        "seg_fig": None,
        "adjustment_results": None,
        "preview_limit": 2000,
        "obs_csv": None,
        "obs_pdf": None,
        "params_csv": None,
        "n": None,
        "u": None,
        "dof": None,
        "hard_constraints": {},
        "soft_constraints": {},
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

init_state()

# ============================== Helper Functions ==============================
def reset_all():
    """Clears all session state variables and re-initializes."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()

# ================================ Layout =====================================
st.set_page_config(page_title="GeoNet Adjust", layout="wide")

# Sidebar for navigation and control
if st.session_state.sidebar_open:
    with st.sidebar:
        pages = ["data input", "Data validation", "adjustment", "visualization", "Download"]
        setup_navigation_buttons(pages, st.session_state.current_step, st.session_state.steps_done)
        st.markdown("---")
        st.button("Clear Data", on_click=reset_all, use_container_width=True)

# Main content area
MAIN = st.container()

with MAIN:
    if st.session_state.dimension is None:
        main_page()
        st.session_state.sidebar_open = True
    elif st.session_state.current_step == "data input":
        data_input_page()
    elif st.session_state.current_step == "Data validation":
        data_validation_page()
    elif st.session_state.current_step == "adjustment":
        adjustment_page()
    elif st.session_state.current_step == "visualization":
        visualization_page()
    elif st.session_state.current_step == "Download":
        download_page()