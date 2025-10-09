import math
import numpy as np

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F_INV = 298.257223563  # Inverse flattening
WGS84_B = WGS84_A * (1 - 1 / WGS84_F_INV)  # Semi-minor axis


def cartesian_to_curvilinear(x, y, z):
    """
    Converts Cartesian coordinates (X, Y, Z) to Curvilinear (Geodetic)
    coordinates (phi, lambda, height) using an iterative method.
    Based on WGS84 ellipsoid.

    Returns:
        tuple: (phi_rad, lambda_rad, height) in radians and meters.
    """
    f = 1 / WGS84_F_INV
    e_sq = 2 * f - f ** 2

    lambda_rad = math.atan2(y, x)

    p = math.sqrt(x ** 2 + y ** 2)
    if p == 0:  # Handle pole case
        phi_rad = math.pi / 2 if z >= 0 else -math.pi / 2
        h = z - WGS84_B if z >= 0 else -z - WGS84_B
        return phi_rad, lambda_rad, h

    # Iterative calculation for latitude
    phi_rad = math.atan(z / (p * (1 - e_sq)))

    for _ in range(10):  # Iterate a few times for convergence
        N = WGS84_A / math.sqrt(1 - e_sq * math.sin(phi_rad) ** 2)
        h = (p / math.cos(phi_rad)) - N
        phi_new = math.atan(z / (p * (1 - (e_sq * N / (N + h)))))
        if abs(phi_new - phi_rad) < 1e-12:
            break
        phi_rad = phi_new

    # Final height calculation
    N = WGS84_A / math.sqrt(1 - e_sq * math.sin(phi_rad) ** 2)
    h = p / math.cos(phi_rad) - N

    return phi_rad, lambda_rad, h


def jacobian_carti_to_curvi(x, y, z, phi_rad):
    """
    Calculates the Jacobian matrix for the transformation from Cartesian to
    Curvilinear coordinates.
    """
    f = 1 / WGS84_F_INV
    e_sq = 2 * f - f ** 2
    N = WGS84_A / math.sqrt(1 - e_sq * math.sin(phi_rad) ** 2)

    # Pre-calculate common terms
    sin_phi = math.sin(phi_rad)
    cos_phi = math.cos(phi_rad)
    sin_lambda = math.sin(math.atan2(y, x))
    cos_lambda = math.cos(math.atan2(y, x))

    R = math.sqrt(x ** 2 + y ** 2)
    if R == 0: R = 1e-6  # Avoid division by zero at the poles

    # Partial derivatives (elements of the Jacobian)
    dphi_dx = - (sin_phi * cos_lambda) / (N * (1 - e_sq))
    dphi_dy = - (sin_phi * sin_lambda) / (N * (1 - e_sq))
    dphi_dz = cos_phi / (N * (1 - e_sq))

    dlambda_dx = -sin_lambda / R
    dlambda_dy = cos_lambda / R
    dlambda_dz = 0

    dh_dx = cos_phi * cos_lambda
    dh_dy = cos_phi * sin_lambda
    dh_dz = sin_phi

    j = np.array([
        [dphi_dx, dphi_dy, dphi_dz],
        [dlambda_dx, dlambda_dy, dlambda_dz],
        [dh_dx, dh_dy, dh_dz]
    ])
    return j


def variance_propagator(C_cartesian, jacobian):
    """Propagates the covariance using the Jacobian."""
    return jacobian @ C_cartesian @ jacobian.T


def radians_to_dms(rad):
    """Converts radians to a DMS string."""
    deg = abs(math.degrees(rad))
    degrees = int(deg)
    minutes_float = (deg - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    sign = "-" if rad < 0 else ""
    return f"{sign}{degrees}° {minutes}' {seconds:.4f}\""


def calculate_geodetic_coordinates(final_results, values, constants, param_names, dimension):
    """
    Main function to calculate geodetic coordinates for all stations.
    It now uses the 'values' dict for initial coordinates and handles 'constants' for fixed stations.
    """
    final_geodetic_params = {}
    initial_geodetic_params = {}

    if dimension != '3D':
        return initial_geodetic_params, final_geodetic_params

    # --- BUG FIX: Convert list of sympy symbols to a list of strings ---
    # This is the critical change to make all lookups succeed.
    param_names_str = [str(p) for p in param_names]

    if not param_names_str:
        return initial_geodetic_params, final_geodetic_params

    stations = sorted(list(set(p.split('_')[1] for p in param_names_str if '_' in p)))
    C_cartesian_full = final_results.get("Sigma_X_hat_Aposteriori", np.array([]))

    # Create a reliable map from parameter name to its index in the X_hat vector
    param_to_index = {name: i for i, name in enumerate(param_names_str)}

    for station in stations:
        sym_x, sym_y, sym_z = f'X_{station}', f'Y_{station}', f'Z_{station}'

        # --- 1. Process Initial Parameters (from 'values' dict) ---
        x_init = values.get(sym_x, 0)
        y_init = values.get(sym_y, 0)
        z_init = values.get(sym_z, 0)

        phi_rad_init, lam_rad_init, h_init = cartesian_to_curvilinear(x_init, y_init, z_init)
        initial_geodetic_params[station] = {
            'phi': radians_to_dms(phi_rad_init),
            'lambda': radians_to_dms(lam_rad_init),
            'h': f"{h_init:.4f}"
        }

        # --- 2. Process Final Parameters (checking for constants) ---
        if sym_x in constants and sym_y in constants and sym_z in constants:
            final_geodetic_params[station] = {
                'phi': ("FIXED", "FIXED"),
                'lambda': ("FIXED", "FIXED"),
                'h': ("FIXED", "FIXED")
            }
        else:
            try:
                # Use the reliable map to get indices
                idx_x = param_to_index[sym_x]
                idx_y = param_to_index[sym_y]
                idx_z = param_to_index[sym_z]
                indices = [idx_x, idx_y, idx_z]

                C_station = C_cartesian_full[np.ix_(indices, indices)]

                x = final_results['X Hat (Final)'][idx_x]
                y = final_results['X Hat (Final)'][idx_y]
                z = final_results['X Hat (Final)'][idx_z]

                phi_rad, lam_rad, h = cartesian_to_curvilinear(x, y, z)

                J = jacobian_carti_to_curvi(x, y, z, phi_rad)
                C_geodetic = variance_propagator(C_station, J)

                sd_phi_rad, sd_lam_rad, sd_h = np.sqrt(np.diag(C_geodetic))

                final_geodetic_params[station] = {
                    'phi': (radians_to_dms(phi_rad), f"{math.degrees(sd_phi_rad) * 3600:.4f}\""),
                    'lambda': (radians_to_dms(lam_rad), f"{math.degrees(sd_lam_rad) * 3600:.4f}\""),
                    'h': (f"{h:.4f}", f"{sd_h:.4f}")
                }
            except (KeyError, IndexError, ValueError):
                final_geodetic_params[station] = {}

        print(initial_geodetic_params, final_geodetic_params)

    return initial_geodetic_params, final_geodetic_params