# batch_adjustment.py
import numpy as np
import sympy as sp
from backend.adjustment.helpers import safe_inverse, get_param_symbols, substitute_and_evaluate


def batch_adjustment(
        L_observed,
        observations_eq,
        initial_guess_vector,
        initial_guess_dict,
        constants,
        P,
        params,
        labels,
        apriori_reference_var=1,
        max_iterations=10,
        tolerance=1e-9,
        constraint_type="No Constraint"
):
    """
    Computes the results of the non-linear observation equation method using iterative least squares.

    Args:
        L_observed (np.ndarray): The observed values of the baselines.
        observations_eq (list): List of sympy expressions for observation equations.
        initial_guess_vector (np.ndarray): The initial guess of the unknown parameters.
        initial_guess_dict (dict): Dictionary mapping sympy symbols to their initial values.
        constants (dict): Dictionary of constant values (e.g., for hard constraints).
        params (dict): A list of the unknown parameters (sympy symbols).
        labels (list): A list of labels for the observations.
        apriori_reference_var (float, optional): The apriori reference variance. Defaults to 1.
        max_iterations (int, optional): The maximum number of iterations. Defaults to 10.
        tolerance (float, optional): The tolerance for convergence. Defaults to 1e-9.
        constraint_type (str, optional): The type of constraint applied ("No Constraint", "Hard Constraint", "Soft Constraint").
                                        Defaults to "No Constraint".

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of the final adjustment results.
            - list: A list of VTPV values for plotting convergence.
    """
    sigma_not_square = apriori_reference_var
    X_hat = initial_guess_vector
    values = initial_guess_dict
    vtpv_values = []
    dof = int(len(observations_eq) - len(X_hat))

    # --- Call helper function ---
    params_symbols = get_param_symbols(params)

    # --- Call helper function ---
    L_not = substitute_and_evaluate(observations_eq, values, constants)
    L = L_observed - L_not

    # Determine if pseudo-inverse should be forced based on constraint type
    force_pinv_flag = (constraint_type == "No Constraint")
    for iteration in range(max_iterations):
        A = sp.Matrix([[obs.diff(param) for param in params_symbols] for obs in observations_eq])
        A_evaluated = np.array(A.subs({**values, **constants}).evalf(), dtype=np.float64)

        N = A_evaluated.T @ P @ A_evaluated
        U = A_evaluated.T @ P @ L

        # --- Use the new safe_inverse helper function ---
        N_inv = safe_inverse(N, force_pinv=force_pinv_flag)
        delta_X = N_inv @ U

        V_linear = A_evaluated @ delta_X - L
        X_hat = X_hat + delta_X
        values = {param: X_hat[i, 0] for i, param in enumerate(params_symbols)}

        # --- Call helper function ---
        L_adjusted = substitute_and_evaluate(observations_eq, values, constants)
        V_non_linear = L_observed.reshape(-1, 1) - L_adjusted

        vtpv = (V_linear.T @ P @ V_linear).item()
        vtpv_values.append(vtpv)

        if np.abs(np.abs(V_non_linear.T @ P @ V_non_linear) - np.abs(V_linear.T @ P @ V_linear)) < tolerance:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break

        L = L_observed.reshape(-1, 1) - L_adjusted

    # Final calculations using the already computed N_inv
    V_final = V_linear
    sigma_not_hat_squared = (V_final.T @ P @ V_final) / dof
    print("Sigma not squared:", sigma_not_hat_squared)

    # --- Use safe_inverse for P matrix as well (though it's typically invertible) ---
    P_inv = safe_inverse(P)
    sigma_VV = sigma_not_hat_squared * (P_inv - A_evaluated @ N_inv @ A_evaluated.T)
    sigma_lhat_lhat = sigma_not_hat_squared * (A_evaluated @ N_inv @ A_evaluated.T)
    sigma_x_hat_apriori = sigma_not_square * N_inv
    sigma_x_hat_aposteriori = sigma_not_hat_squared * N_inv

    # 1. Define a threshold for ill-conditioning. 1e7 is a reasonable value.
    CONDITION_THRESHOLD = 1e7

    # 2. Check the condition number of the normal matrix N and print a warning if it's high.
    Warning = ''
    if np.linalg.cond(N) > CONDITION_THRESHOLD:
        Warning +=("\n⚠️ WARNING: The normal matrix (N) is ill-conditioned.\n"
                   " This indicates a weak network geometry (e.g., collinear points or poor redundancy).\n"
                   " The results may be numerically unstable.\n"
                   " INFO: Negative variances found on the diagonal of sigma_VV due to instability. Clamping these values to zero to allow calculations to proceed.")
        print("\n⚠️ WARNING: The normal matrix (N) is ill-conditioned.")
        print("   This indicates a weak network geometry (e.g., collinear points or poor redundancy).")
        print("   The results may be numerically unstable.\n")

    # 3. Correct the diagonal of the sigma_VV matrix.
    # Get a copy of the diagonal to check for negative values.
    diag_sigma_VV = np.diagonal(sigma_VV).copy()

    # Check if any diagonal elements are negative.
    if np.any(diag_sigma_VV < 0):
        print("INFO: Negative variances found on the diagonal of sigma_VV due to instability.")
        print("      Clamping these values to zero to allow calculations to proceed.")


        # Use np.maximum to replace any element less than 0 with 0.
        corrected_diag = np.maximum(10**(-6), diag_sigma_VV)
        # Place the corrected diagonal back into the sigma_VV matrix.
        np.fill_diagonal(sigma_VV, corrected_diag)
    # --- End: Your program can now safely use the corrected sigma_VV ---

    first_check = np.abs(np.abs(V_final.T @ P @ V_final) - np.abs(-V_final.T @ P @ L)) < 10 ** (-6)
    second_check = np.all(np.abs(A_evaluated.T @ P @ V_final) < 10 ** (-6))

    final_results = {
        "Equations": observations_eq, "N": len(observations_eq), "P Matrix": P,
        "Design Matrix (Final)": A_evaluated, "Normal Matrix": N, "Residuals": V_final,
        "L Adjusted": L_adjusted, "L Observed": L_observed, "Apriori Variance": sigma_not_square,
        "Aposteriori Variance": sigma_not_hat_squared, "X Hat (Final)": X_hat, "Delta_X": delta_X,
        "Sigma_VV": sigma_VV, "Sigma L Adjusted": sigma_lhat_lhat, "Sigma_X_hat_Apriori": sigma_x_hat_apriori,
        "Sigma_X_hat_Aposteriori": sigma_x_hat_aposteriori, "Iterations": iteration + 1,
        "First Check Passed V.T@P@V = -V.T@P@L": first_check,
        "Second Check Passed A.T@P@V = 0": second_check, "DOF": dof, "PARAMS_Name": params_symbols,
        "Constant": constants, "Labels": labels, "VTPV_values": vtpv_values,
    }
    return final_results, vtpv_values,Warning
