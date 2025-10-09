import streamlit as st
from utils.adjustment_helpers import (
    hard_constraint_ui,
    soft_constraint_ui,
    get_rejection_level_from_alpha
)
from backend.processing_pipeline import run_full_pipeline


def adjustment_page():
    """
    Renders the Adjustment page, using the correct st.dialog syntax for modern Streamlit versions.
    """

    # --- DIALOG FOR PROCESSING ---
    if st.session_state.get("show_processing_modal"):

        # This is the correct way to use st.dialog in modern versions.
        @st.dialog("⚙️ Processing Analysis", width="small")
        def run_analysis_dialog():
            # Add custom styling for a clean, compact dialog
            st.markdown("""
            <style>
            .dialog-container {
                text-align: center;
                padding: 1.5rem;
                border-radius: 15px;
                background: #f9f9f9;
                box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
                margin: 0 auto;
            }
            .dialog-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #2E7D32;
                margin-bottom: 1rem;
            }
            .dialog-subtext {
                font-size: 1rem;
                color: #555;
                margin-bottom: 1.2rem;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="dialog-container">', unsafe_allow_html=True)

            # If processing is still running
            if not st.session_state.get("processing_complete"):
                st.markdown('<div class="dialog-header">🔄 Running Full Pipeline</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="dialog-subtext">Please wait while we process your data.<br>Do not close this window.</div>',
                    unsafe_allow_html=True)

                with st.spinner("⏳ Processing..."):
                    try:
                        pipeline_results = run_full_pipeline(st.session_state)
                        st.session_state.update(pipeline_results)
                    except Exception as e:
                        st.error(f"❌ An error occurred during analysis:\n\n{e}")
                        st.session_state.processing_complete = True
                st.rerun()

            # If processing is done
            else:
                st.markdown('<div class="dialog-header">✅ Analysis Complete!</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="dialog-subtext">Your data has been successfully processed.<br>You can now close this dialog and continue.</div>',
                    unsafe_allow_html=True)

                if st.button("Close", use_container_width=True, key="close_modal"):
                    st.session_state.show_processing_modal = False
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # Call the function to display the dialog
        run_analysis_dialog()

    # --- Main Page UI (renders when the dialog is not shown) ---
    else:
        st.header("📐 Adjustment Settings")
        st.markdown(
            "Configure your network adjustment, then run the full analysis to generate all results, plots, and reports in one go."
        )

        if not st.session_state.get("steps_done", {}).get("Data validation", False):
            st.info("⚠️ Please complete **Data Validation** first.")
            return

        # --- UI for Adjustment Settings (Expanders) ---
        with st.expander("⚙️ Adjustment Procedure", expanded=True):
            st.session_state.adjustment_type = st.selectbox(
                "Select Adjustment Method", ["Batch Adjustment", "Phased Adjustment"],
                index=["Batch Adjustment", "Phased Adjustment"].index(
                    st.session_state.get("adjustment_type", "Batch Adjustment")),
                help="Choose between standard batch least squares or phased adjustment (multi-block)."
            )
            if st.session_state.adjustment_type == "Phased Adjustment":
                st.session_state.block_size = st.number_input(
                    "Block Size", value=20, step=1, help="Set block size for phased adjustment."
                )
                st.info("ℹ️ Phased adjustment logic is not fully implemented yet.")

        with st.expander("🧮 Weight Matrix Settings", expanded=True):
            st.session_state.weight_matrix = st.selectbox(
                "Choose Weight Matrix Type", ["Unity", "Full", "Diagonal"],
                index=["Unity", "Full", "Diagonal"].index(st.session_state.get("weight_matrix", "Unity")),
                key="weight_matrix_select", help="Specify how observation weights are defined."
            )

        with st.expander("🔗 Constraint Settings", expanded=True):
            st.session_state.constraint_type = st.selectbox(
                "Constraint Type", ["Free Net Adjustment", "Hard Constraint", "Soft Constraint"],
                index=["Free Net Adjustment", "Hard Constraint", "Soft Constraint"].index(
                    st.session_state.get("constraint_type", "Free Net Adjustment")),
                key="constraint_type_select", help="Fix points or apply conditions to the solution."
            )
            if st.session_state.constraint_type == "Free Net Adjustment":
                st.session_state.hard_constraints = {}
                st.session_state.soft_constraints = {}
            elif st.session_state.constraint_type == "Hard Constraint":
                st.info("🔒 Define hard constraints (fixed points or values).")
                hard_constraint_ui()
            elif st.session_state.constraint_type == "Soft Constraint":
                st.info("🪶 Define soft constraints (weighted conditions).")
                soft_constraint_ui(st.session_state.weight_matrix)

        with st.expander("🔬 Statistical Testing & Outlier Detection", expanded=True):
            st.markdown("###### Statistical Parameters")
            col_alpha, col_beta, col_rejection = st.columns(3)
            with col_alpha:
                st.session_state.alpha = st.number_input(
                    "Significance Level (α)", 0.001, 0.100,
                    value=st.session_state.get("alpha", 0.001), step=0.001, format="%.3f",
                    help="Probability of Type I error (rejecting a good observation)."
                )
            with col_beta:
                st.session_state.beta_power = st.number_input(
                    "Desired Power (1-β)", 0.50, 0.99,
                    value=st.session_state.get("beta_power", 0.80), step=0.01, format="%.2f",
                    help="Probability of correctly detecting a blunder."
                )
            rejection_level = get_rejection_level_from_alpha(st.session_state.alpha)
            st.session_state.rejection_level = rejection_level

            st.markdown("###### Blunder Detection")
            st.session_state.blunder_detection_method = st.selectbox(
                "Blunder Detection Method", ["None", "Baarda Data Snooping", "Tau Test"],
                index=["None", "Baarda Data Snooping", "Tau Test"].index(
                    st.session_state.get("blunder_detection_method", "None")),
                help="Choose a statistical test for detecting outliers after adjustment."
            )
            if st.session_state.blunder_detection_method == "Baarda Data Snooping":
                with col_rejection:
                    st.metric(
                        label="Rejection Level (k)", value=f"{rejection_level:.3f}",
                        help="Standardized residuals greater than this will be flagged as outliers."
                    )
            if st.session_state.blunder_detection_method == "Tau Test":
                st.info("The Tau Test will be available in a future release.")

        # --- BUTTONS to Run Analysis and Navigate ---
        st.markdown("---")
        st.subheader("🚀 Run Full Analysis")
        st.markdown("Click the button below to run the entire analysis pipeline.")

        if st.button("✅ Run Full Analysis & Generate All Results", use_container_width=True, type="primary"):
            st.session_state.show_processing_modal = True
            if "processing_complete" in st.session_state:
                del st.session_state.processing_complete
            st.rerun()

        if st.session_state.get("processing_complete"):
            st.success("Analysis is complete. You may now navigate to the other pages using the buttons below.")

        st.markdown("---")
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("⬅️ Previous", use_container_width=True, key="prev_adj"):
                st.session_state.current_step = "Data validation"
                st.rerun()
        with next_col:
            if st.button("Next ➡️", use_container_width=True, key="next_adj",
                         disabled=not st.session_state.get("processing_complete")):
                st.session_state.current_step = "visualization"
                st.rerun()