import numpy as np
import sympy as sp
from backend.adjustment.helpers import safe_inverse, get_param_symbols, substitute_and_evaluate


def iterative_outlier_detection(initial_result, rejection_level, labels, params,force_pinv=True):
    """
    Performs iterative outlier detection on a set of observations using the Baarda Data Snooping method.

    This function repeatedly identifies and removes the single largest outlier from a dataset until
    no observations are flagged as outliers or the degrees of freedom are zero. The process works
    as follows:
    1. Calculate the standardized residuals for all observations.
    2. Identify the observation with the largest standardized residual.
    3. If this residual exceeds a specified rejection level, it's flagged as an outlier.
    4. The outlier is removed, and the least squares adjustment is recalculated without it.
    5. The results of this new adjustment are stored, and the process repeats from step 1.
    6. The loop terminates if no outliers are found in an iteration or if DOF <= 0.

    Args:
        initial_result (dict): A dictionary containing the results from the initial least squares adjustment.
                               This should include keys like 'Residuals', 'P Matrix', 'Design Matrix (Final)',
                               'Aposteriori Variance', 'Sigma_VV', 'DOF', etc.
        rejection_level (float): The critical value from the standard normal distribution corresponding
                                 to a chosen significance level (e.g., 3.29 for alpha = 0.001).
                                 An observation is considered an outlier if its standardized residual
                                 exceeds this value.
        labels (list or np.ndarray): A list of labels corresponding to the observations in the
                                     initial_result. This list is updated as outliers are removed.
        params (dict): A dictionary mapping station names to their SymPy parameter symbols,
                       used for recalculating symbolic expressions.
        force_pinv (bool): If True, forces the use of pseudo-inverse for the
                               Normal matrix, which is common for free-network adjustments
                               often associated with outlier detection. Defaults to True.

    Returns:
        list: A list of dictionaries. Each dictionary represents one iteration of outlier removal
              and contains:
              - All the calculated adjustment results for that iteration.
              - 'removed_index': The original index of the observation removed in that step.
    """
    outlier_detection_results = []
    curr_result = initial_result.copy()
    curr_labels = list(labels)
    vtpv_values = curr_result.get("VTPV_values", [])

    # --- Call helper function ---
    params_symbols = get_param_symbols(params)
    original_indices = list(range(len(initial_result["Equations"])))

    while True:
        if curr_result["DOF"] <= 0:
            print("Stopping: Degrees of freedom are zero or less.")
            break

        curr_V = curr_result['Residuals']
        curr_P = curr_result['P Matrix']
        curr_A = curr_result['Design Matrix (Final)']
        curr_delta_X = curr_result['Delta_X']
        curr_N = curr_result['Normal Matrix']
        curr_X_hat = curr_result['X Hat (Final)']
        curr_Q_XX = curr_result['Sigma_X_hat_Apriori'] / curr_result['Apriori Variance']
        cofactor_VV = curr_result['Sigma_VV'] / curr_result['Aposteriori Variance']
        curr_sigma_not_sq = curr_result['Aposteriori Variance']
        constants = curr_result["Constant"]

        max_residual_val = 0
        idx_to_remove = []
        for i in range(len(curr_V)):
            test_value = abs(curr_V[i, 0]) / np.sqrt(cofactor_VV[i, i])
            if test_value > (np.sqrt(curr_sigma_not_sq) * rejection_level):
                if test_value > max_residual_val:
                    max_residual_val = test_value
                    idx_to_remove = [i]

        if not idx_to_remove:
            print("Stopping: No more outliers found.")
            break

        blunder_local_idx = idx_to_remove[0]
        original_removed_index = original_indices.pop(blunder_local_idx)

        blunder_eq = curr_result["Equations"][blunder_local_idx]
        blunder_values = {param: val for param, val in zip(params_symbols, curr_X_hat)}
        blunder_A_sp = sp.Matrix([[obs.diff(param) for param in params_symbols] for obs in [blunder_eq]])
        A_blunder = np.array(blunder_A_sp.subs({**blunder_values, **constants}).evalf(), dtype=np.float64)
        P_blunder = np.array([[curr_P[blunder_local_idx, blunder_local_idx]]])

        # --- Use the new safe_inverse helper function ---
        N_inv = safe_inverse(curr_N, force_pinv=force_pinv)
        P_blunder_inv = safe_inverse(P_blunder)
        M_inv = safe_inverse(P_blunder_inv - A_blunder @ N_inv @ A_blunder.T)

        delta_corr_x_hat = N_inv @ A_blunder.T @ M_inv @ (A_blunder @ curr_delta_X - curr_V[blunder_local_idx])
        X_hat_adj = curr_X_hat + delta_corr_x_hat

        delta_corr_Q_xx = -1 * (N_inv @ A_blunder.T @ M_inv @ A_blunder @ N_inv)
        Q_xx_adj = curr_Q_XX + delta_corr_Q_xx

        new_P = np.delete(np.delete(curr_P, blunder_local_idx, axis=0), blunder_local_idx, axis=1)
        new_A = np.delete(curr_A, blunder_local_idx, axis=0)
        new_L_obs = np.delete(curr_result["L Observed"], blunder_local_idx, axis=0)
        new_equations = np.delete(curr_result["Equations"], blunder_local_idx, axis=0)
        del curr_labels[blunder_local_idx]

        V_adj = new_A @ delta_corr_x_hat + np.delete(curr_V, blunder_local_idx, axis=0)
        vtpv = (V_adj.T @ new_P @ V_adj).item()
        vtpv_values.append(vtpv)

        new_dof = curr_result["DOF"] - 1
        aposteriori_updated = vtpv / new_dof if new_dof > 0 else 0

        new_N = new_A.T @ new_P @ new_A
        new_N_inv = safe_inverse(new_N, force_pinv=force_pinv)
        new_P_inv = safe_inverse(new_P)

        sigma_x_hat_adj_aposteriori = aposteriori_updated * Q_xx_adj
        sigma_x_hat_adj_apriori = curr_result["Apriori Variance"] * Q_xx_adj
        sigma_vv_adjusted = aposteriori_updated * (new_P_inv - new_A @ new_N_inv @ new_A.T)
        new_L_adjusted = new_L_obs - V_adj

        iteration_results = {
            "removed_index": original_removed_index, "Equations": new_equations, "N": len(new_equations),
            "P Matrix": new_P, "Design Matrix (Final)": new_A, "Normal Matrix": new_N,
            "Residuals": V_adj, "L Adjusted": new_L_adjusted, "L Observed": new_L_obs,
            "Apriori Variance": curr_result["Apriori Variance"], "Aposteriori Variance": aposteriori_updated,
            "X Hat (Final)": X_hat_adj, "Delta_X": delta_corr_x_hat, "Sigma_VV": sigma_vv_adjusted,
            "Sigma L Adjusted": (aposteriori_updated * new_P_inv) - sigma_vv_adjusted,
            "Sigma_X_hat_Apriori": sigma_x_hat_adj_apriori,
            "Sigma_X_hat_Aposteriori": sigma_x_hat_adj_aposteriori, "DOF": new_dof,
            "PARAMS_Name": curr_result["PARAMS_Name"], "Constant": constants, "Labels": list(curr_labels),
        }

        outlier_detection_results.append(iteration_results)
        curr_result = iteration_results

    return outlier_detection_results, vtpv_values