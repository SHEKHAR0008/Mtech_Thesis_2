import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


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
        params (list): A list of the unknown parameters (sympy symbols).
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

    # Store results for each iteration
    vtpv_values = []
    dof = int(len(observations_eq) - len(X_hat))

    def get_param_symbols(params):
        """Flatten {station: (X, Y, Z)} into a list of sympy symbols."""
        symbols = []
        for sta, (X_sym, Y_sym, Z_sym) in params.items():
            symbols.extend([X_sym, Y_sym, Z_sym])
        return symbols

    params_symbols = get_param_symbols(params)

    def substitute_and_evaluate(observations_eq, values, constants):
        """Substitute values into observations and evaluate to numerical results."""
        L_not = np.array(
            [obs.subs({**values, **constants}).evalf() for obs in observations_eq],
            dtype=np.float64,
        ).reshape(-1, 1)
        return L_not

    L_not = substitute_and_evaluate(observations_eq, values, constants)
    L = L_observed - L_not

    for iteration in range(max_iterations):
        # Compute design matrix A
        A = sp.Matrix(
            [[obs.diff(param) for param in params_symbols] for obs in observations_eq]
        )

        # Substitute current values into A and evaluate
        A_evaluated = np.array(A.subs({**values, **constants}).evalf(), dtype=np.float64)

        # Compute normal matrix N and observation vector U
        N = A_evaluated.T @ P @ A_evaluated
        U = A_evaluated.T @ P @ L

        # Solve for delta_X based on constraint type
        try:
            if constraint_type == "No Constraint":
                # Use pseudo-inverse for rank-deficient systems (no constraints)
                delta_X = np.linalg.pinv(N) @ U
            else:
                # Use standard inverse for well-conditioned systems (with constraints)
                delta_X = np.linalg.inv(N) @ U
        except np.linalg.LinAlgError:
            print("Error: Singular matrix encountered in iteration", iteration + 1)
            break

        V_linear = A_evaluated @ delta_X - L

        # Update X_hat with delta_X
        X_hat = X_hat + delta_X

        # Update values for next iteration
        values = {param: X_hat[i, 0] for i, param in enumerate(params_symbols)}

        # Recompute L_not with updated X_hat and calculate non-linear residuals
        L_adjusted = substitute_and_evaluate(observations_eq, values, constants)
        V_non_linear = L_observed.reshape(-1, 1) - L_adjusted

        # Store V.T P V for convergence plot
        vtpv = (V_linear.T @ P @ V_linear).item()
        vtpv_values.append(vtpv)

        # Check convergence
        if np.abs(np.abs(V_non_linear.T @ P @ V_non_linear) - np.abs(V_linear.T @ P @ V_linear)) < tolerance:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break

        # Re-calculate L for next iteration
        L = L_observed.reshape(-1, 1) - L_adjusted

    # Final calculations after loop termination
    V_final = V_linear
    sigma_not_hat_squared = (V_final.T @ P @ V_final) / dof
    sigma_VV = sigma_not_hat_squared * (np.linalg.inv(P) - A_evaluated @ np.linalg.pinv(N) @ A_evaluated.T)
    sigma_lhat_lhat = sigma_not_hat_squared * (A_evaluated @ np.linalg.pinv(N) @ A_evaluated.T)
    sigma_x_hat_apriori = sigma_not_square * np.linalg.pinv(N)
    sigma_x_hat_aposteriori = sigma_not_hat_squared * np.linalg.pinv(N)

    # Final checks
    first_check = np.abs(np.abs(V_final.T @ P @ V_final) - np.abs(-V_final.T @ P @ L)) < 10 ** (-6)
    second_check = np.all(np.abs(A_evaluated.T @ P @ V_final) < 10 ** (-6))

    final_results = {
        "Equations": observations_eq,
        "N":len(observations_eq),
        "P Matrix": P,
        "Design Matrix (Final)": A_evaluated,
        "Residuals": V_final,
        "L Adjusted": L_adjusted,
        "L Observed": L_observed,
        "Apriori Variance": sigma_not_square,
        "Aposteriori Variance": sigma_not_hat_squared,
        "X Hat (Final)": X_hat,
        "Delta_X": delta_X,
        "Sigma_VV": sigma_VV,
        "Sigma L Adjusted": sigma_lhat_lhat,
        "Sigma_X_hat_Apriori": sigma_x_hat_apriori,
        "Sigma_X_hat_Aposteriori": sigma_x_hat_aposteriori,
        "Iterations": iteration + 1,
        "First Check Passed V.T@P@V = -V.T@P@L": first_check,
        "Second Check Passed A.T@P@V = 0": second_check,
        "DOF": dof,
        "PARAMS_Name": params_symbols,
        "Constant": constants,
        "Labels": labels,
        "VTPV_values": vtpv_values,  # Added this for graph generation
    }

    return final_results, vtpv_values