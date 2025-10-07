import streamlit as st
import numpy as np
from backend.adjustment.observation_equation import build_observation_system
from backend.adjustment.initial_guess import initial_guess
from backend.adjustment.batch_adjustment import batch_adjustment
from backend.adjustment.apply_constraint_fn import apply_constraints
from backend.adjustment.outlier_detection import iterative_outlier_detection
from utils.adjustment_helpers import (
    hard_constraint_ui,
    soft_constraint_ui,
    stringify_keys,
    preview_matrix,
    get_rejection_level_from_alpha
)


def adjustment_page():
    """
    Renders the Adjustment page content with a professional, structured, and user-friendly UI.
    """
    st.header("📐 Adjustment Settings")
    st.markdown(
        "Configure and run network adjustment with constraints, weight matrices, and outlier detection."
    )

    if not st.session_state["steps_done"]["Data validation"]:
        st.info("⚠️ Please complete **Data Validation** first.")
        return

    # ----------------------------
    # 1️⃣ Adjustment Procedure
    # ----------------------------
    with st.expander("⚙️ Adjustment Procedure", expanded=True):
        st.session_state.adjustment_type = st.selectbox(
            "Select Adjustment Method",
            ["Batch Adjustment", "Phased Adjustment"],
            index=["Batch Adjustment", "Phased Adjustment"].index(
                st.session_state.get("adjustment_type", "Batch Adjustment")
            ),
            help="Choose between standard batch least squares or phased adjustment (multi-block)."
        )

        if st.session_state.adjustment_type == "Phased Adjustment":
            st.session_state.block_size = st.number_input(
                "Block Size", value=20, step=1,
                help="Set block size for phased adjustment (number of observations per block)."
            )
            st.info("ℹ️ Phased adjustment logic is not fully implemented yet.")

    # ----------------------------
    # 2️⃣ Weight Matrix Selection
    # ----------------------------
    with st.expander("🧮 Weight Matrix Settings", expanded=True):
        st.session_state.weight_matrix = st.selectbox(
            "Choose Weight Matrix Type",
            ["Unity", "Full", "Diagonal"],
            index=["Unity", "Full", "Diagonal"].index(
                st.session_state.get("weight_matrix", "Unity")
            ),
            key="weight_matrix_select",
            help="Specify how observation weights are defined in the adjustment."
        )

    # ----------------------------
    # 3️⃣ Constraint Options
    # ----------------------------
    with st.expander("🔗 Constraint Settings", expanded=True):
        st.session_state.constraint_type = st.selectbox(
            "Constraint Type",
            ["No Constraint", "Hard Constraint", "Soft Constraint"],
            index=["No Constraint", "Hard Constraint", "Soft Constraint"].index(
                st.session_state.get("constraint_type", "No Constraint")
            ),
            key="constraint_type_select",
            help="Constraints allow you to fix points or apply conditions to the solution."
        )

        if st.session_state.constraint_type == "No Constraint":
            st.session_state.hard_constraints = {}
            st.session_state.soft_constraints = {}
        elif st.session_state.constraint_type == "Hard Constraint":
            st.info("🔒 Define hard constraints (fixed points or values).")
            hard_constraint_ui()
        elif st.session_state.constraint_type == "Soft Constraint":
            st.info("🪶 Define soft constraints (weighted conditions).")
            soft_constraint_ui(st.session_state.weight_matrix)

    # ----------------------------
    # 4️⃣ Statistical Testing
    # ----------------------------
    with st.expander("🔬 Statistical Testing ", expanded=True):
        st.markdown("###### Statistical Parameters")
        col_alpha, col_beta, col_rejection = st.columns(3)
        with col_alpha:
            st.session_state.alpha = st.number_input(
                "Significance Level (α)",
                min_value=0.001, max_value=0.100,
                value=st.session_state.get("alpha", 0.001),
                step=0.001, format="%.3f",
                help="Probability of Type I error (rejecting a good observation)."
            )
        with col_beta:
            st.session_state.beta_power = st.number_input(
                "Desired Power (1-β)",
                min_value=0.50, max_value=0.99,
                value=st.session_state.get("beta_power", 0.80),
                step=0.01, format="%.2f",
                help="Probability of correctly detecting a blunder."
            )

        rejection_level = get_rejection_level_from_alpha(st.session_state.alpha)
        st.session_state.rejection_level = rejection_level


    # ----------------------------
    # 4️⃣ Outlier Detection
    # ----------------------------
    with st.expander(" Outlier Detection", expanded=True):
        st.session_state.blunder_detection_method = st.selectbox(
            "Blunder Detection Method",
            ["None", "Baarda Data Snooping", "Tau Test"],
            index=["None", "Baarda Data Snooping", "Tau Test"].index(
                st.session_state.get("blunder_detection_method", "None")
            ),
            help="Choose a statistical test for detecting outliers after adjustment."
        )

        if st.session_state.blunder_detection_method == "Baarda Data Snooping":

            with col_rejection:
                st.metric(
                    label="Rejection Level (k)",
                    value=f"{rejection_level:.3f}",
                    help="Standardized residuals greater than this will be flagged as outliers."
                )

        if st.session_state.blunder_detection_method == "Tau Test":
            st.info("The Tau Test will be available in a future release.")



    # ----------------------------
    # 5️⃣ Run Adjustment
    # ----------------------------
    st.markdown("---")
    st.subheader("🚀 Run Adjustment")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Submit and View Results", use_container_width=True):
            run_adjustment_logic()
    with col2:
        if st.button("➡️ Submit and Next", use_container_width=True):
            run_adjustment_logic(go_next=True)

    # ----------------------------
    # 📊 Display Results (if available)
    # ----------------------------
    if st.session_state["steps_done"]["adjustment"] and st.session_state.viewed_results:
        st.success("✅ Adjustment complete!")
        # ----------------------------
        # 📊 Summary Panel
        # ----------------------------
        with st.expander("📊 Adjustment Summary (Live Preview)", expanded=True):
            st.markdown("This panel shows a live summary of your current configuration.")

            # Read values safely from session_state
            adj_type = st.session_state.get("adjustment_type", "Batch Adjustment")
            weight = st.session_state.get("weight_matrix", "Unity")
            constraint = st.session_state.get("constraint_type", "No Constraint")
            blunder = st.session_state.get("blunder_detection_method", "None")
            alpha_val = st.session_state.get("alpha", None)
            beta_val = st.session_state.get("beta_power", None)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Adjustment Method", adj_type)
                st.metric("Weight Matrix", weight)
            with col2:
                st.metric("Constraint Type", constraint)
                st.metric("Blunder Detection", blunder)
            with col3:
                st.metric("α (Significance Level)", f"{alpha_val:.3f}")
                st.metric("Desired Power (1-β)", f"{beta_val:.3f}")

        with st.expander("📊 System Dimensions", expanded=True):
            final_results = st.session_state.final_results
            st.write(f"**Number of observations (N):** {final_results['N']}")
            st.write(f"**Number of unknowns (U):** {final_results['N'] - final_results['DOF']}")
            st.write(f"**Degrees of freedom (dof):** {final_results['DOF']}")

        if len(st.session_state.warning) != 0:
            st.warning(st.session_state.warning)

        st.subheader("📈 Variances")
        st.metric("Apriori Variance (σ₀²)", f"{final_results.get('Apriori Variance', 'N/A'):.6f}")
        val = final_results.get("Aposteriori Variance", None)

        def safe_format(value, digits=6):
            if value is None:
                return "N/A"
            if isinstance(value, (int, float, np.floating)):
                return f"{value:.{digits}f}"
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return f"{value.item():.{digits}f}"
                else:
                    return f"{np.mean(value):.{digits}f} (mean)"
            return str(value)

        st.metric("Aposteriori Variance (σ̂₀²)", safe_format(val))

        st.subheader("✅ Adjustment Checks")
        col1_check, col2_check = st.columns(2)
        with col1_check:
            if final_results.get("First Check Passed V.T@P@V = -V.T@P@L"):
                st.success("First Check Passed ✅")
            else:
                st.error("First Check Failed ❌")
        with col2_check:
            if final_results.get("Second Check Passed A.T@P@V = 0"):
                st.success("Second Check Passed ✅")
            else:
                st.error("Second Check Failed ❌")

        st.markdown("---")
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("⬅️ Previous", use_container_width=True, key="prev_adj"):
                st.session_state.current_step = "Data validation"
                st.session_state.viewed_results = False
                st.session_state["steps_done"]["adjustment"] = False
                st.session_state.final_results = None
                st.rerun()
        with next_col:
            if st.button("Next ➡️", use_container_width=True, key="next_adj"):
                st.session_state.current_step = "visualization"
                st.session_state.viewed_results = False
                st.rerun()


@st.cache_resource(show_spinner="Running adjustment...")
def run_adjustment_logic(go_next=False):
    try:
        obs_vec, equations, params, labels, P, n, u, dof = build_observation_system(
            st.session_state.baseline_list,
            weight_type=st.session_state.get("weight_matrix", "unity").lower()
        )

        obs_vec, equations, P, labels, n, u, dof = apply_constraints(
            obs_vec, equations, params, P, labels,
            hard_constraints=st.session_state.get("hard_constraints", None),
            soft_constraints=st.session_state.get("soft_constraints", None),
            weight_type=st.session_state.weight_matrix.lower()
        )

        values, X_hat, constants, params = initial_guess(
            st.session_state.baseline_list,
            params,
            hard_constraints=st.session_state.get("hard_constraints", None),
            soft_constraints=st.session_state.get("soft_constraints", None)
        )

        final_results, vtpv_values, warning = batch_adjustment(
            obs_vec, equations, X_hat, values, constants, P, params, labels,
            apriori_reference_var=1, max_iterations=10, tolerance=1e-9,
            constraint_type=st.session_state.get("constraint_type", "No Constraint")
        )

        if st.session_state.blunder_detection_method == "Baarda Data Snooping":
            outlier_results, vtpv_values = iterative_outlier_detection(
                final_results, st.session_state.rejection_level, labels, params, force_pinv=True
            )
            st.session_state.outlier_results = outlier_results

        st.session_state.warning = warning
        st.session_state.final_results = final_results
        st.session_state.vtpv_values = vtpv_values
        st.session_state.dof = final_results["DOF"]
        st.session_state.n = final_results["N"]
        st.session_state.u = final_results["N"] - st.session_state.dof

        st.session_state["steps_done"]["adjustment"] = True
        st.session_state.viewed_results = True

        if go_next:
            st.session_state.current_step = "visualization"
            st.session_state.viewed_results = False

        st.rerun()

    except Exception as e:
        st.error(f"❌ Adjustment failed: {e}")
