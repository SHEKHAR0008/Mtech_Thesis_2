# content/download.py

import streamlit as st
import io
import os
from backend.csv_report.csv_result import export_adjustment_results_excel
from backend.csv_report.report import generate_adjustment_report_html_pdf
from backend.plots import generate_plots


def download_page():
    """
    Renders the Download page, providing options to export adjustment results
    as CSV/Excel files and a comprehensive report.
    """
    st.header("Download")
    if not st.session_state["steps_done"]["adjustment"]:
        st.info("Please complete Adjustment first.")
        return

    # Generate and cache CSVs on first entry to this page
    if "obs_buffer" not in st.session_state:
        st.session_state.obs_buffer, st.session_state.params_buffer = export_adjustment_results_excel(
            st.session_state.final_results,
            st.session_state.outlier_results,
            st.session_state.blunder_detection_method,
            st.session_state.alpha,
            st.session_state.beta_power,
            rejection_level=st.session_state.rejection_level,
            geodetic_coords=None,
            initial_results= st.session_state.final_results,
        )

    # ---------------- CSV EXPORTS ----------------
    st.subheader("CSV Exports")
    csv_cols = st.columns(2, gap="large")
    with csv_cols[0]:
        if st.session_state.obs_buffer:
            st.download_button(
                "📊 Download Observations Excel",
                st.session_state.obs_buffer,
                file_name="observations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.button("📊 Download Observations Excel", disabled=True, use_container_width=True)

    with csv_cols[1]:
        if st.session_state.params_buffer:
            st.download_button(
                "📊 Download Parameters Excel",
                st.session_state.params_buffer,
                file_name="params.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.button("📊 Download Parameters Excel", disabled=True, use_container_width=True)

    # ---------------- Report Export ----------------
    st.subheader("Report Export")

    # NEW LOGIC: Check for and generate plots if they don't exist in session state
    if "chi_graph" not in st.session_state or st.session_state.chi_graph is None:
        try:
            plots = generate_plots(
                st.session_state.final_results,
                st.session_state.dof,
                st.session_state.unq_stations,
                st.session_state.baseline_list,
            )
            st.session_state.chi_graph = plots.get("chi_square_plot")
            st.session_state.vtpv_graph = plots.get("vtpv_plot")
            # Note: Network and error ellipse plots are not required for the report,
            # but we can store them for consistency.
            st.session_state.network_plot = plots.get("network_plot")
            st.session_state.error_ellipse_plot = plots.get("error_ellipse_plot")
            st.session_state.error_ellipse_stats = plots.get("error_ellipse_stats")
        except Exception as e:
            st.error(f"Failed to generate plots for report: {e}")
            st.session_state.chi_graph = None
            st.session_state.vtpv_graph = None
            st.session_state.network_plot = None
            st.session_state.error_ellipse_plot = None
            st.session_state.error_ellipse_stats = None

    # Generate and cache the PDF/DOCX report
    if "report_buffer" not in st.session_state:
        try:

            st.session_state.report_buffer = generate_adjustment_report_html_pdf(
                final_results=st.session_state.final_results,
                template_full_path="D:/GeoNet_SRC_CD/backend/csv_report/report_template.html",
                hard_constraints=st.session_state.get("hard_constraints"),
                soft_constraints=st.session_state.get("soft_constraints"),
                vtpv_graph=st.session_state.get("vtpv_graph"),
                chi_graph=st.session_state.get("chi_graph"),
                weight_type=st.session_state.get("weight_matrix", "Unity"),
                adjust_method=st.session_state.get("adjustment_type", "Batch Adjustment"),
                blunder_test=st.session_state.get("blunder_detection_method", "None"),
                alpha=st.session_state.get("alpha", None),
                beta=st.session_state.get("beta_power", None),
                # rejection_level=None,
                # geodetic_coords=None,
                initial_results={},
                network_plot=st.session_state.network_plot,
                error_ellipse_plots = st.session_state.error_ellipse_plot,
                error_ellipse_stats = st.session_state.error_ellipse_stats
            )
        except Exception as e:
            st.error(f"Failed to generate report: {e}")
            st.session_state.report_buffer = None

    if st.session_state.report_buffer:
        st.download_button(
            "⬇️ Download Adjustment Report (DOCX)",
            st.session_state.report_buffer,
            file_name="adjustment_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.button("⬇️ Download Adjustment Report (DOCX)", disabled=True, use_container_width=True)

    st.markdown("---")
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("⬅️ Previous", use_container_width=True, key="prev_down"):
            st.session_state.current_step = "visualization"
            st.session_state.viewed_results = False
            st.session_state.report_buffer = None
            st.session_state.obs_buffer = None
            st.session_state.params_buffer = None
            st.rerun()
    with next_col:
        # Note: 'Download' is the last step in the current workflow.
        st.button("Next ➡️", disabled=True, use_container_width=True)