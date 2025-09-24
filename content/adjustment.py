# content/adjustment.py

import streamlit as st
import numpy as np
from backend.adjustment.observation_equation import build_observation_system
from backend.adjustment.initial_guess import initial_guess
from backend.adjustment.batch_adjustment import batch_adjustment
from backend.adjustment.apply_constraint_fn import apply_constraints
from backend.adjustment.outlier_detection import iterative_outlier_detection
from utils.adjustment_helpers import (hard_constraint_ui, soft_constraint_ui,
                                      stringify_keys, preview_matrix,
                                      get_rejection_level_from_alpha)


def adjustment_page():
    """
    Renders the Adjustment page content.
    """
    st.header("Adjustment")
    if not st.session_state["steps_done"]["Data validation"]:
        st.info("Please complete Data Validation first.")
        return

    st.subheader("Choose Adjustment Procedure")
    st.session_state.adjustment_type = st.selectbox(
        "Adjustment Procedure",
        ["Batch Adjustment", "Phased Adjustment"],
        index=["Batch Adjustment", "Phased Adjustment"].index(
            st.session_state.get("adjustment_type", "Batch Adjustment")
        ),
    )

    if st.session_state.adjustment_type == "Batch Adjustment":
        # --- Reordered Section: Weight Matrix first ---
        st.subheader("Select Weight Matrix")
        st.session_state.weight_matrix = st.selectbox(
            "Weight Matrix",
            ["Unity", "Full", "Diagonal"],
            index=["Unity", "Full", "Diagonal"].index(
                st.session_state.get("weight_matrix", "Unity")
            ),
            key="weight_matrix_select"
        )

        st.subheader("Choose Type of Constraint")
        constraint_type = st.selectbox(
            "Constraint Type",
            ["No Constraint", "Hard Constraint", "Soft Constraint"],
            index=["No Constraint", "Hard Constraint", "Soft Constraint"].index(
                st.session_state.get("constraint_type", "No Constraint")
            ),
            key="constraint_type_select"
        )
        st.session_state.constraint_type = constraint_type

        # Check if a weight matrix is selected before allowing constraints
        if constraint_type in ["Hard Constraint", "Soft Constraint"] and st.session_state.get("weight_matrix", "Unity") is None:
            st.warning("Please select a Weight Matrix first.")
            st.session_state.constraint_type = "No Constraint"
            constraint_type = "No Constraint" # Reset constraint type to avoid confusion

        if constraint_type == "No Constraint":
            st.session_state.hard_constraints = {}
            st.session_state.soft_constraints = {}
        elif constraint_type == "Hard Constraint":
            # Pass the weight type to hard constraint UI if needed, though current logic doesn't use it
            hard_constraint_ui()
        elif constraint_type == "Soft Constraint":
            # Dynamically render soft constraint UI based on selected weight type
            soft_constraint_ui(st.session_state.weight_matrix)
    else:
        st.subheader("Phased Adjustment Settings")
        st.session_state.block_size = st.number_input("Block Size", value=20, step=1)
        st.info("Phased adjustment logic is not fully implemented in the provided code.")

    with st.expander("🔬 Statistical Testing & Outlier Detection", expanded=True):
        st.session_state.blunder_detection_method = st.selectbox(
            "Blunder Detection Method", ["None", "Baarda Data Snooping", "Tau Test"],
            index=["None", "Baarda Data Snooping", "Tau Test"].index(
                st.session_state.get("blunder_detection_method", "None")),
            help="Choose a statistical method to detect and remove outliers after the initial adjustment."
        )
        if st.session_state.blunder_detection_method == "Baarda Data Snooping":
            st.markdown("###### Set Statistical Parameters")
            col_alpha, col_beta, col_rejection = st.columns(3)
            with col_alpha:
                st.session_state.alpha = st.number_input(
                    "Significance Level (α)", min_value=0.001, max_value=0.100,
                    value=st.session_state.get("alpha", 0.001), step=0.001, format="%.3f",
                    help="Probability of Type I error (rejecting a good observation). A common stringent value is 0.001."
                )
            with col_beta:
                st.session_state.beta_power = st.number_input(
                    "Desired Power of Test (1-β)", min_value=0.50, max_value=0.99,
                    value=st.session_state.get("beta_power", 0.80), step=0.01, format="%.2f",
                    help="Probability of correctly detecting a blunder. Used to analyze test sensitivity (e.g., MDB)."
                )

            # --- LOGIC TO CALCULATE AND STORE REJECTION LEVEL ---
            # Call the helper function to get the rejection level
            rejection_level = get_rejection_level_from_alpha(st.session_state.alpha)
            # Store it in the session state so other functions can access it
            st.session_state.rejection_level = rejection_level

            with col_rejection:
                st.metric(
                    label="Rejection Level (k)",
                    value=f"{rejection_level:.3f}",
                    help="Calculated critical value. Standardized residuals greater than this will be flagged as outliers."
                )

        elif st.session_state.blunder_detection_method == "Tau Test":
            st.info("The Tau Test is not yet implemented.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit and View Results", use_container_width=True):
            run_adjustment_logic()
    with col2:
        if st.button("Submit and Next", use_container_width=True):
            run_adjustment_logic(go_next=True)

    # Conditional display of results
    if st.session_state["steps_done"]["adjustment"] and st.session_state.viewed_results:
        st.success("Adjustment complete!")

        with st.expander("📊 System Dimensions", expanded=True):
            final_results = st.session_state.final_results
            st.write(f"**Number of observations (N):** {final_results['N']}")
            st.write(f"**Number of unknowns (U):** {final_results['N'] - final_results['DOF']}")
            st.write(f"**Degrees of freedom (dof):** {final_results['DOF']}")

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
                st.success("First Check Passed")
            else:
                st.error("First Check Failed")
        with col2_check:
            if final_results.get("Second Check Passed A.T@P@V = 0"):
                st.success("Second Check Passed")
            else:
                st.error("Second Check Failed")

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
        # Step 1: Build observation system
        obs_vec, equations, params, labels, P, n, u, dof = build_observation_system(
            st.session_state.baseline_list,
            weight_type=st.session_state.get("weight_matrix", "unity").lower()
        )
        print("Initial")
        # Step 2: Apply constraints
        obs_vec, equations, P, labels, n, u, dof = apply_constraints(
            obs_vec, equations, params, P, labels,
            hard_constraints=st.session_state.get("hard_constraints", None),
            soft_constraints=st.session_state.get("soft_constraints", None),
            weight_type=st.session_state.weight_matrix.lower()
        )
        print("Initaoal")

        # Step 3: Get initial guess
        values, X_hat, constants, params = initial_guess(
            st.session_state.baseline_list,
            params,
            hard_constraints=st.session_state.get("hard_constraints", None),
            soft_constraints=st.session_state.get("soft_constraints", None)
        )
        print("Final")
        # Step 4: Run batch adjustment
        final_results, vtpv_values = batch_adjustment(
            obs_vec, equations, X_hat, values, constants, P, params, labels,
            apriori_reference_var=1, max_iterations=10, tolerance=1e-9,
            constraint_type=st.session_state.get("constraint_type", "No Constraint")
        )

        # Step 5: Run outlier detection if asked
        if st.session_state.blunder_detection_method == "Baarda Data Snooping":
            outlier_results,vtpv_values = iterative_outlier_detection(
                final_results, st.session_state.rejection_level, labels, params, force_pinv=True)

            st.session_state.outlier_results = outlier_results

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
        st.error(f"Adjustment failed: {e}")