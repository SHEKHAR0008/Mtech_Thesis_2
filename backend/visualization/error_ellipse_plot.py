import numpy as np
import plotly.graph_objects as go
import sympy
from scipy.stats import chi2


def plot_interactive_error_ellipses(final_results, conf=0.95):
    """
    Generate interactive Plotly plots for error ellipses (2D), ellipsoids (3D),
    or line intervals (1D), skipping fixed stations. Returns plots and stats.

    Args:
        final_results (dict): Geodetic adjustment results.
        conf (float): Confidence level (0.95 by default).

    Returns:
        plots (list): List of Plotly figures (one per station)
        stats (dict): Dictionary with ellipse/ellipsoid/interval parameters
    """
    plots = []
    stats = {}

    try:
        X_hat_final = final_results.get("X Hat (Final)")
        params_names = final_results.get("PARAMS_Name")
        covariance_matrix = final_results.get("Sigma_X_hat_Aposteriori")
        constants = final_results.get("Constant", {})

        if X_hat_final is None or params_names is None or covariance_matrix is None:
            print("Required data not found in final_results dictionary.")
            return plots, stats

        # Identify constant stations
        constant_stations = set()
        for const_symbol in constants.keys():
            # Use string manipulation that's robust to symbol names
            station_name = str(const_symbol).split('_', 1)[1]
            constant_stations.add(station_name)

        # Organize station coordinates and indices
        station_coords = {}
        for i, param_symbol in enumerate(params_names):
            symbol_str = str(param_symbol)
            coord_type, station_name = symbol_str.split("_", 1)
            if station_name not in station_coords:
                station_coords[station_name] = {"coords": {}, "indices": {}}
            station_coords[station_name]["coords"][coord_type] = X_hat_final[i, 0]
            station_coords[station_name]["indices"][coord_type] = i

        # Loop over stations to generate plots
        for station_name, data in station_coords.items():
            if station_name in constant_stations:
                continue

            present_coords = sorted(data["coords"].keys())
            dim = len(present_coords)

            if dim == 0:
                continue

            # --- 2D Case (Error Ellipse) ---
            if dim == 2:
                idx_x, idx_y = data["indices"]["X"], data["indices"]["Y"]
                # FIX: Explicitly convert the covariance sub-matrix to float
                C = np.array(covariance_matrix[np.ix_([idx_x, idx_y], [idx_x, idx_y])], dtype=float)
                print(C)

                try:
                    eig_vals, eig_vecs = np.linalg.eig(C)
                except np.linalg.LinAlgError:
                    continue  # Skip if matrix is invalid

                if not np.all(eig_vals > 0):
                    continue  # Skip if not positive semi-definite

                # Sort eigenvalues and eigenvectors from largest to smallest
                order = np.argsort(eig_vals)[::-1]
                eig_vals = eig_vals[order]
                eig_vecs = eig_vecs[:, order]

                k = np.sqrt(chi2.ppf(conf, 2))
                a, b = k * np.sqrt(eig_vals)  # a = semi-major axis, b = semi-minor
                theta = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

                stats[station_name] = {"type": "2D", "a": a, "b": b, "theta_deg": theta}

                t = np.linspace(0, 2 * np.pi, 200)
                ellipse = np.array([a * np.cos(t), b * np.sin(t)])
                rotated = eig_vecs @ ellipse

                center_x, center_y = data["coords"]["X"], data["coords"]["Y"]
                x_ellipse = rotated[0, :] + center_x
                y_ellipse = rotated[1, :] + center_y

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode="lines", name=f"{station_name} Ellipse"))
                fig.add_trace(go.Scatter(
                    x=[center_x], y=[center_y], mode="markers+text",
                    marker=dict(color="red", size=8),
                    text=[f"{station_name}<br>a={a:.4f}, b={b:.4f}, θ={theta:.1f}°"],
                    textposition="top center", name=station_name
                ))
                fig.update_layout(title=f"2D Error Ellipse for {station_name} ({conf * 100:.0f}%)",
                                  xaxis_title="X", yaxis_title="Y", width=600, height=600, yaxis_scaleanchor="x")
                plots.append(fig)

            # --- 3D case ---
            elif dim == 3:
                idxs = [data["indices"][c] for c in ["X", "Y", "Z"]]
                # FIX: Ensure the covariance sub-matrix is float
                C = np.array(covariance_matrix[np.ix_(idxs, idxs)], dtype=float)

                try:
                    eig_vals, eig_vecs = np.linalg.eig(C)
                except np.linalg.LinAlgError:
                    continue

                if not np.all(eig_vals > 0):
                    continue

                k = np.sqrt(chi2.ppf(conf, 3))
                radii = k * np.sqrt(eig_vals)
                stats[station_name] = {"type": "3D", "radii": sorted(list(map(float, radii)), reverse=True)}

                u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
                x = radii[0] * np.cos(u) * np.sin(v)
                y = radii[1] * np.sin(u) * np.sin(v)
                z = radii[2] * np.cos(v)

                points = np.stack([x.flatten(), y.flatten(), z.flatten()])
                coords_rotated = eig_vecs @ points

                center = [data["coords"]["X"], data["coords"]["Y"], data["coords"]["Z"]]
                x_final = coords_rotated[0, :].reshape(x.shape) + center[0]
                y_final = coords_rotated[1, :].reshape(y.shape) + center[1]
                z_final = coords_rotated[2, :].reshape(z.shape) + center[2]

                fig = go.Figure()
                fig.add_trace(go.Surface(x=x_final, y=y_final, z=z_final, opacity=0.5,
                                         colorscale="Viridis", showscale=False))
                fig.add_trace(go.Scatter3d(
                    x=[center[0]], y=[center[1]], z=[center[2]],
                    mode="markers+text", text=[station_name],
                    marker=dict(size=5, color="red")
                ))
                fig.update_layout(title=f"3D Error Ellipsoid for {station_name} ({conf * 100:.0f}%)",
                                  scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='data'))
                plots.append(fig)

    except Exception as e:
        print(f"Error generating plots: {e}")
        # Add import traceback; traceback.print_exc() here for more detailed debugging
        return [], {}

    return plots, stats

