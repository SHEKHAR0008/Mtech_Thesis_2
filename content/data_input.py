# content/data_input.py

import streamlit as st
import platform
import psutil

def truncate_text(s: str, limit: int) -> str:
    """Helper to truncate long text for preview."""
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[Truncated]"

def data_input_page():
    """
    Renders the Data Input page content.
    """
    st.header("Data Input")

    uploaded = st.file_uploader("Upload Baseline File (.txt/.xlsx/.xls)", type=["txt", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if uploaded.type == "text/plain" or uploaded.name.lower().endswith(".txt"):
                content = uploaded.read().decode("utf-8", errors="ignore")
            else:
                content = uploaded.name + " [Uploaded Excel file. Paste or use your parser accordingly]"
            st.session_state.baseline_text = content
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    st.session_state.baseline_text = st.text_area(
        "Baseline Data (editable)",
        value=st.session_state.baseline_text,
        height=200,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit and View Results", use_container_width=True):
            if st.session_state.baseline_text.strip():
                st.session_state["steps_done"]["data input"] = True
                st.session_state.viewed_results = True
                st.rerun()
            else:
                st.error("Please provide baseline data.")
    with col2:
        if st.button("Submit and Next", use_container_width=True):
            if st.session_state.baseline_text.strip():
                st.session_state["steps_done"]["data input"] = True
                st.session_state.current_step = "Data validation"
                st.session_state.viewed_results = False
                st.rerun()
            else:
                st.error("Please provide baseline data.")

    if st.session_state["steps_done"]["data input"] and st.session_state.viewed_results:
        st.subheader("Data Preview")
        st.code(truncate_text(st.session_state.baseline_text, st.session_state.preview_limit))
        st.success("Data input complete.")

        st.markdown("---")
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.current_step = "Data validation"
            st.session_state.viewed_results = False
            st.rerun()