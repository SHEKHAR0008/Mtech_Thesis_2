import numpy as np
import plotly.graph_objects as go
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
            print("❌ Required data not found in final_results dictionary.")
            return plots, stats

        # Identify constant stations (fixed points)
        constant_stations = set()
        for const_symbol in constants.keys():
            station_name = str(const_symbol).split('_', 1)[1]
            constant_stations.add(station_name)

        # Organize station coordinates and their indices
        station_coords = {}
        for i, param_symbol in enumerate(params_names):
            symbol_str = str(param_symbol)
            coord_type, station_name = symbol_str.split("_", 1)
            if station_name not in station_coords:
                station_coords[station_name] = {"coords": {}, "indices": {}}
            station_coords[station_name]["coords"][coord_type] = X_hat_final[i, 0]
            station_coords[station_name]["indices"][coord_type] = i

        # Loop through stations
        for station_name, data in station_coords.items():
            if station_name in constant_stations:
                continue  # Skip fixed stations

            present_coords = sorted(data["coords"].keys())
            dim = len(present_coords)
            if dim == 0:
                continue

            # ✅ --- 2D Error Ellipse ---
            if dim == 2:
                idx_x, idx_y = data["indices"]["X"], data["indices"]["Y"]
                C = np.array(covariance_matrix[np.ix_([idx_x, idx_y], [idx_x, idx_y])], dtype=float)

                # Eigen-decomposition
                eig_vals, eig_vecs = np.linalg.eig(C)
                eig_vals = np.real(eig_vals)
                if not np.all(eig_vals > 0):
                    continue

                order = np.argsort(eig_vals)[::-1]
                eig_vals = eig_vals[order]
                eig_vecs = eig_vecs[:, order]

                k = np.sqrt(chi2.ppf(conf, 2))
                a, b = k * np.sqrt(eig_vals)
                theta = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

                stats[station_name] = {"type": "2D", "a": a, "b": b, "theta_deg": theta}

                # Generate ellipse
                t = np.linspace(0, 2 * np.pi, 200)
                ellipse = np.array([a * np.cos(t), b * np.sin(t)])
                rotated = eig_vecs @ ellipse

                center_x, center_y = data["coords"]["X"], data["coords"]["Y"]
                x_ellipse = rotated[0, :] + center_x
                y_ellipse = rotated[1, :] + center_y

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_ellipse, y=y_ellipse, mode="lines", name=f"{station_name} Ellipse"
                ))
                fig.add_trace(go.Scatter(
                    x=[center_x], y=[center_y], mode="markers+text",
                    marker=dict(color="red", size=8),
                    text=[f"{station_name}<br>a={a:.4f}, b={b:.4f}, θ={theta:.1f}°"],
                    textposition="top center", name=station_name
                ))

                fig.update_layout(
                    title=f"2D Error Ellipse for {station_name} ({conf * 100:.0f}%)",
                    xaxis_title="X",
                    yaxis_title="Y",
                    width=600,
                    height=600,
                    yaxis_scaleanchor="x"
                )
                plots.append(fig)

                # --- 3D Error Ellipsoid ---
            elif dim == 3:
                idxs = [data["indices"][c] for c in ["X", "Y", "Z"]]
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

                # The station's true center coordinate (can be very large)
                center_coord = np.array([data["coords"]["X"], data["coords"]["Y"], data["coords"]["Z"]])

                # --- SOLUTION: Calculate the ellipsoid shape around (0,0,0) ---
                # We will NOT add the large 'center_coord' to the points.
                u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
                x = radii[0] * np.cos(u) * np.sin(v)
                y = radii[1] * np.sin(u) * np.sin(v)
                z = radii[2] * np.cos(v)

                points = np.stack([x.flatten(), y.flatten(), z.flatten()])
                coords_rotated = eig_vecs @ points

                x_final = coords_rotated[0, :].reshape(x.shape)
                y_final = coords_rotated[1, :].reshape(y.shape)
                z_final = coords_rotated[2, :].reshape(z.shape)

                fig = go.Figure()
                # Plot the ellipsoid surface, now centered at the origin
                fig.add_trace(go.Surface(x=x_final, y=y_final, z=z_final, opacity=0.5,
                                         colorscale="Viridis", showscale=False, name="Ellipsoid"))

                # Plot the station's center point, also at the origin
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode="markers", text=[station_name],
                    marker=dict(size=5, color="red"),
                    name="Station Center"
                ))

                # Update the title to include the true global coordinates as text
                fig.update_layout(
                    title=f"3D Error Ellipsoid for {station_name} ({conf * 100:.0f}%)"
                          f"<br><sup>True Center (X,Y,Z): {np.round(center_coord, 3)}</sup>",
                    scene=dict(
                        xaxis_title=f"dX (m) from {station_name}",
                        yaxis_title=f"dY (m) from {station_name}",
                        zaxis_title=f"dZ (m) from {station_name}",
                        aspectmode='data'  # This keeps the shape correct
                    )
                )
                plots.append(fig)

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        return [], {}

    return plots, stats
