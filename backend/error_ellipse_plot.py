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
            coord_type, station_name = str(const_symbol).split("_", 1)
            constant_stations.add(station_name)

        # Organize station coordinates
        station_coords = {}
        for i, param_symbol in enumerate(params_names):
            symbol_str = str(param_symbol)
            coord_type, station_name = symbol_str.split("_", 1)
            if station_name not in station_coords:
                station_coords[station_name] = {"X": None, "Y": None, "Z": None, "indices": {}}
            station_coords[station_name][coord_type] = X_hat_final[i, 0]
            station_coords[station_name]["indices"][coord_type] = i

        # Loop over stations
        for station_name, data in station_coords.items():
            if station_name in constant_stations:
                continue

            present_coords = [c for c in ["X", "Y", "Z"] if data[c] is not None]
            dim = len(present_coords)

            if dim == 0:
                continue

            # --- 1D case ---
            if dim == 1:
                coord = present_coords[0]
                idx = data["indices"][coord]
                var = covariance_matrix[idx, idx]
                if var <= 0:
                    continue

                k = np.sqrt(chi2.ppf(conf, 1))
                sigma = np.sqrt(var)
                interval = k * sigma

                stats[station_name] = {"type": "1D", "coord": coord, "interval": float(interval)}

                # Plot
                if coord == "X":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[data["X"] - interval, data["X"] + interval],
                        y=[0, 0], mode="lines+markers",
                        name=f"{station_name} (X ± {interval:.3f})"
                    ))
                    fig.update_layout(title=f"1D Confidence Interval for {station_name} ({conf*100:.0f}%)",
                                      xaxis_title="X", yaxis_title="")
                elif coord == "Y":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[0, 0],
                        y=[data["Y"] - interval, data["Y"] + interval],
                        mode="lines+markers",
                        name=f"{station_name} (Y ± {interval:.3f})"
                    ))
                    fig.update_layout(title=f"1D Confidence Interval for {station_name} ({conf*100:.0f}%)",
                                      xaxis_title="", yaxis_title="Y")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[0, 0],
                        y=[data["Z"] - interval, data["Z"] + interval],
                        mode="lines+markers",
                        name=f"{station_name} (Z ± {interval:.3f})"
                    ))
                    fig.update_layout(title=f"1D Confidence Interval for {station_name} ({conf*100:.0f}%)",
                                      xaxis_title="", yaxis_title="Z")

                plots.append(fig)

            # --- 2D case ---
            elif dim == 2:
                idx_x, idx_y = data["indices"]["X"], data["indices"]["Y"]
                C = covariance_matrix[np.ix_([idx_x, idx_y], [idx_x, idx_y])]
                if not np.all(np.linalg.eigvals(C) > 0):
                    continue

                k = np.sqrt(chi2.ppf(conf, 2))
                eig_vals, eig_vecs = np.linalg.eig(C)
                a, b = k * np.sqrt(eig_vals)
                order = np.argsort([a, b])[::-1]
                a, b = [a, b][order[0]], [a, b][order[1]]
                theta = np.degrees(np.arctan2(eig_vecs[1, order[0]], eig_vecs[0, order[0]]))

                stats[station_name] = {"type": "2D", "a": float(a), "b": float(b), "theta_deg": float(theta)}

                t = np.linspace(0, 2*np.pi, 200)
                ellipse = np.array([a*np.cos(t), b*np.sin(t)])
                rotated = eig_vecs @ ellipse
                x_ellipse = rotated[0, :] + data["X"]
                y_ellipse = rotated[1, :] + data["Y"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode="lines", name=f"{station_name} Ellipse"))
                fig.add_trace(go.Scatter(
                    x=[data["X"]], y=[data["Y"]],
                    mode="markers+text",
                    marker=dict(color="red", size=8),
                    text=[f"{station_name}<br>a={a:.3f}, b={b:.3f}, θ={theta:.1f}°"],
                    textposition="top center",
                    name=station_name
                ))
                fig.update_layout(title=f"2D Error Ellipse for {station_name} ({conf*100:.0f}%)",
                                  xaxis_title="X", yaxis_title="Y", width=600, height=600)
                plots.append(fig)

            # --- 3D case ---
            elif dim == 3:
                idxs = [data["indices"][c] for c in ["X","Y","Z"]]
                C = covariance_matrix[np.ix_(idxs, idxs)]
                if not np.all(np.linalg.eigvals(C) > 0):
                    continue

                k = np.sqrt(chi2.ppf(conf, 3))
                eig_vals, eig_vecs = np.linalg.eig(C)
                radii = k * np.sqrt(eig_vals)

                stats[station_name] = {"type": "3D", "radii": list(map(float, radii))}

                # Surface mesh
                u = np.linspace(0, 2*np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                x = radii[0]*np.outer(np.cos(u), np.sin(v))
                y = radii[1]*np.outer(np.sin(u), np.sin(v))
                z = radii[2]*np.outer(np.ones_like(u), np.cos(v))
                coords = np.dot(np.array([x.flatten(), y.flatten(), z.flatten()]).T, eig_vecs)
                x_final = coords[:,0].reshape(x.shape) + data["X"]
                y_final = coords[:,1].reshape(y.shape) + data["Y"]
                z_final = coords[:,2].reshape(z.shape) + data["Z"]

                fig = go.Figure()
                fig.add_trace(go.Surface(x=x_final, y=y_final, z=z_final, opacity=0.5,
                                         colorscale="Viridis", showscale=False))
                fig.add_trace(go.Scatter3d(
                    x=[data["X"]], y=[data["Y"]], z=[data["Z"]],
                    mode="markers+text", text=[station_name],
                    marker=dict(size=5, color="red")
                ))
                fig.update_layout(title=f"3D Error Ellipsoid for {station_name} ({conf*100:.0f}%)",
                                  scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
                plots.append(fig)

    except Exception as e:
        print(f"Error generating plots: {e}")
        return [], {}

    return plots, stats
