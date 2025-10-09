# content/download.py
import streamlit as st

def download_page():
    st.header("⬇️ Download Results")
    if not st.session_state.get("steps_done", {}).get("Download", False):
        st.info("Please run the full analysis from the 'Adjustment' page first.")
        return

    st.subheader("CSV Exports")
    csv_cols = st.columns(2, gap="large")
    with csv_cols[0]:
        st.download_button(
            "📊 Download Observations Excel",
            st.session_state.get("obs_buffer", b""),
            file_name="observations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            disabled=not st.session_state.get("obs_buffer")
        )
    with csv_cols[1]:
        st.download_button(
            "📊 Download Parameters Excel", st.session_state.params_buffer,
            file_name="params.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, disabled=not st.session_state.params_buffer
        )

    st.subheader("Report Export")
    report_cols = st.columns(2, gap="large")
    with report_cols[0]:
        st.download_button(
            "📄 Download PDF Report",
            st.session_state.get("report_buffer", b""),
            file_name="Adjustment_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=not st.session_state.get("report_buffer")
        )
    with report_cols[1]:
        st.download_button(
            "📊 Download Var-Covar Excel", st.session_state.covar_buffer,
            file_name="covar.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, disabled=not st.session_state.covar_buffer
        )

    st.markdown("---")
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("⬅️ Previous", use_container_width=True, key="prev_down"):
            st.session_state.current_step = "visualization"
            st.rerun()
    with next_col:
        st.button("Next ➡️", disabled=True, use_container_width=True)