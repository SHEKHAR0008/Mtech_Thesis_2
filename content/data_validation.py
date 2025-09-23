# content/data_validation.py

import streamlit as st
from backend.data_io import parse_baseline_text
from backend.loop_check import check_all_loops
from utils.ui_helpers import setup_navigation_buttons

def data_validation_page():
    """
    Renders the Data Validation page content.
    """
    st.header("Data Validation")
    if not st.session_state["steps_done"]["data input"]:
        st.info("Please complete Data Input first.")
        return

    # Input for threshold
    threshold = st.number_input(
        "Set Misclosure Threshold (meters)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="A loop is considered 'closed' if its misclosure is below this value."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit and View Results", use_container_width=True):
            run_validation_logic(threshold)
    with col2:
        if st.button("Submit and Next", use_container_width=True):
            run_validation_logic(threshold, go_next=True)

    # Conditional display of results
    if st.session_state["steps_done"]["Data validation"] and st.session_state.viewed_results:
        st.subheader("Results Summary")
        st.metric("Threshold Used", f"{threshold:.3f} m")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("#### Baseline Count")
            if st.session_state.baseline_list:
                st.info(f"{len(st.session_state.baseline_list)} baselines detected")
            else:
                st.warning("No baselines detected.")

            st.write("#### Validation Status")
            if st.session_state.loop_ok:
                st.success("✅ All loops successfully closed.")
            else:
                st.error("❌ Loop misclosures detected.")

        with col2:
            st.write("#### Loop Misclosure Details")
            st.text(st.session_state.loop_msg)

        if st.session_state.baseline_list:
            with st.expander("Show All Baselines Detected"):
                for i in st.session_state.baseline_list:
                    st.write(i)

        st.markdown("---")
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("⬅️ Previous", use_container_width=True, key="prev_val"):
                st.session_state.current_step = "data input"
                # Clear this step's data
                st.session_state.viewed_results = False
                st.session_state["steps_done"]["Data validation"] = False
                st.session_state.baseline_list = None
                st.session_state.unq_stations = None
                st.session_state.loop_msg = ""
                st.rerun()
        with next_col:
            if st.button("Next ➡️", use_container_width=True, key="next_val"):
                st.session_state.current_step = "adjustment"
                st.session_state.viewed_results = False
                st.rerun()

def run_validation_logic(threshold, go_next=False):
    """
    Helper function to perform the validation logic and update state.
    """
    try:
        st.session_state.baseline_list, st.session_state.unq_stations = parse_baseline_text(st.session_state.baseline_text)
        ok, msg, _ = check_all_loops(st.session_state.baseline_list, threshold=threshold)
        st.session_state.loop_ok = ok
        st.session_state.loop_msg = msg
        st.session_state["steps_done"]["Data validation"] = True
        st.session_state.viewed_results = True
        if go_next:
            st.session_state.current_step = "adjustment"
            st.session_state.viewed_results = False
        st.rerun()
    except Exception as e:
        st.error(f"Validation failed: {e}")
        st.session_state.loop_ok = False
        st.session_state.loop_msg = f"Validation failed: {e}"
        st.session_state["steps_done"]["Data validation"] = False