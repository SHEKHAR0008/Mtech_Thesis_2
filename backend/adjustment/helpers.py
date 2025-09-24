# helpers.py
import numpy as np
import sympy as sp
from numpy.typing import NDArray

def safe_inverse(matrix: NDArray, force_pinv: bool = False) -> NDArray:
    """
    Computes the inverse of a matrix, with options for robustness.

    Args:
        matrix (NDArray): The matrix to be inverted.
        force_pinv (bool): If True, directly uses the pseudo-inverse.
                           If False, tries the standard inverse first and
                           falls back to the pseudo-inverse on failure.

    Returns:
        NDArray: The inverted matrix.
    """
    if force_pinv:
        return np.linalg.pinv(matrix)
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # print("Warning: Standard inverse failed. Falling back to pseudo-inverse.")
        return np.linalg.pinv(matrix)

def get_param_symbols(params: dict) -> list:
    """
    Flattens a dictionary of station parameters {station: (X, Y, Z)}
    into a single list of sympy symbols.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        list: A flat list of sympy symbols.
    """
    symbols = []
    for _, data in params.items():
        if len(data) == 2:
            X_sym,Y_sym = data
            symbols.extend([X_sym, Y_sym])
        elif len(data) == 3:
            X_sym,Y_sym,Z_sym = data
            symbols.extend([X_sym,Y_sym,Z_sym])
        else:
            symbols.extend(data)
    return symbols

def substitute_and_evaluate(observations_eq: list, values: dict, constants: dict) -> NDArray:
    """
    Substitutes numerical values into sympy observation equations and evaluates them.

    Args:
        observations_eq (list): A list of sympy expressions.
        values (dict): A dictionary of symbol-value pairs for variables.
        constants (dict): A dictionary of symbol-value pairs for constants.

    Returns:
        NDArray: A numpy column vector of the evaluated numerical results.
    """
    L_not = np.array(
        [obs.subs({**values, **constants}).evalf() for obs in observations_eq],
        dtype=np.float64,
    ).reshape(-1, 1)
    return L_not