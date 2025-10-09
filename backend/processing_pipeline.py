# backend/processing_pipeline.py

import streamlit as st

# Import all necessary backend functions
from backend.adjustment.observation_equation import build_observation_system
from backend.adjustment.initial_guess import initial_guess
from backend.adjustment.batch_adjustment import batch_adjustment
from backend.adjustment.apply_constraint_fn import apply_constraints
from backend.adjustment.outlier_detection import iterative_outlier_detection
from backend.plots import generate_plots
from backend.csv_report.csv_result import export_adjustment_results_excel
from backend.csv_report.report import generate_adjustment_report_html_pdf
from backend.geodetic_utils import calculate_geodetic_coordinates

def run_full_pipeline(state):
    """
    Runs the entire data processing pipeline from adjustment to report generation.

    Args:
        state: The st.session_state object containing all inputs.

    Returns:
        A dictionary containing all the calculated results and artifacts.
    """
    results = {}

    # --- 1. Run Network Adjustment ---
    obs_vec, equations, params, labels, P, n, u, dof = build_observation_system(
        state.baseline_list,
        weight_type=state.get("weight_matrix", "unity").lower()
    )
    obs_vec, equations, P, labels, n, u, dof = apply_constraints(
        obs_vec, equations, params, P, labels,
        hard_constraints=state.get("hard_constraints"),
        soft_constraints=state.get("soft_constraints"),
        weight_type=state.weight_matrix.lower()
    )
    values, X_hat, constants, params = initial_guess(
        state.baseline_list, params,
        hard_constraints=state.get("hard_constraints"),
        soft_constraints=state.get("soft_constraints")
    )
    final_results, vtpv_values, warning = batch_adjustment(
        obs_vec, equations, X_hat, values, constants, P, params, labels
    )
    results["final_results"] = final_results
    results["warning"] = warning
    results["dof"] = final_results["DOF"]

    outlier_results = None
    if state.blunder_detection_method == "Baarda Data Snooping":
        outlier_results, _ = iterative_outlier_detection(
            final_results, state.rejection_level, labels, params, force_pinv=True
        )

    results["outlier_results"] = outlier_results

    if outlier_results:
        final_results = outlier_results[-1]

    values = {str(k): v for k, v in values.items()}
    constants = {str(k): v for k, v in constants.items()}
    initial_geodetic, final_geodetic = calculate_geodetic_coordinates(
        final_results,
        values, constants,  # Pass initial parameter guesses
        final_results.get("PARAMS_Name", []),
        state.dimension
    )
    results["initial_geodetic_params"] = initial_geodetic
    results["final_geodetic_params"] = final_geodetic

    # --- 2. Generate Visualization Plots ---
    plots = generate_plots(
        final_results,
        results["dof"],
        state.unq_stations,
        state.baseline_list,
    )
    results.update(plots) # Add all generated plots to the results dict
    # --- 3. Generate Downloadable Files (CSVs and Report) ---
    print("csv")
    obs_buffer, params_buffer, covar_buffer = export_adjustment_results_excel(
        final_results,
        outlier_results,
        state.blunder_detection_method,
        state.alpha,
        state.beta_power,
        initial_geodetic,
        final_geodetic,
        rejection_level=state.rejection_level,
    )
    results["obs_buffer"] = obs_buffer
    results["params_buffer"] = params_buffer
    results["covar_buffer"] = covar_buffer
    print("reporting")
    report_buffer = generate_adjustment_report_html_pdf(
        final_results=final_results,
        template_full_path="D:/GeoNet_SRC_CD/backend/csv_report/report_template.html",
        adjustment_dimension=state.dimension,
        initial_geodetic_params  = initial_geodetic,
        final_geodetic_params  = final_geodetic,
        values = values,
        constants = constants,
        hard_constraints=state.get("hard_constraints"),
        soft_constraints=state.get("soft_constraints"),
        vtpv_graph=results.get("vtpv_plot"),
        chi_graph=results.get("chi_square_plot"),
        weight_type=state.get("weight_matrix", "Unity"),
        adjust_method=state.get("adjustment_type", "Batch Adjustment"),
        blunder_test=state.get("blunder_detection_method", "None"),
        alpha=state.get("alpha"),
        beta=state.get("beta_power"),
        network_plot=results.get("network_plot"),
        error_ellipse_plots = results.get("error_ellipse_plot"),
        error_ellipse_stats = results.get("error_ellipse_stats")
    )
    results["report_buffer"] = report_buffer

    # --- 4. Mark processing as complete ---
    results["steps_done"] = st.session_state["steps_done"]
    results["steps_done"]["adjustment"] = True
    results["steps_done"]["visualization"] = True
    results["steps_done"]["Download"] = True
    results["processing_complete"] = True

    return results