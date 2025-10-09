import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import matplotlib.figure
import os
import plotly.graph_objects as go

def format_matrix_for_report(matrix, max_rows=30, max_cols=14):
    """
    Formats a large matrix for display in a report by truncating it and
    applying scientific notation with 3-digit precision.

    If the matrix is larger than the specified max dimensions, it will be
    truncated with ellipses (...) in the middle of rows and columns. All
    numerical values are formatted to the '{:.3e}' format.

    Args:
        matrix (list | np.ndarray | pd.DataFrame): The matrix to format.
        max_rows (int): The maximum number of rows to display.
        max_cols (int): The maximum number of columns to display.

    Returns:
        str: A formatted string representation of the potentially truncated matrix.
    """
    df = pd.DataFrame(matrix)
    n_rows, n_cols = df.shape

    # Define the formatting function
    formatter = lambda x: f'{x:.3e}' if isinstance(x, (int, float, np.number)) else x

    # If the matrix is small enough, format it and return directly
    if n_rows <= max_rows and n_cols <= max_cols:
        return df.map(formatter).to_string()

    # --- Truncate Rows if necessary ---
    if n_rows > max_rows:
        top_rows = max_rows // 2
        bottom_rows = max_rows - top_rows
        # Create a "..." row to insert between the top and bottom sections
        dots_row = pd.DataFrame([['...'] * n_cols], columns=df.columns, index=['...'])
        df = pd.concat([df.head(top_rows), dots_row, df.tail(bottom_rows)])

    # --- Truncate Columns if necessary ---
    if n_cols > max_cols:
        left_cols = max_cols // 2
        right_cols = max_cols - left_cols

        # Select left and right columns
        df_left = df.iloc[:, :left_cols]
        df_right = df.iloc[:, -right_cols:]

        # Create a "..." column to insert
        dots_col = pd.DataFrame({'...': ['...'] * len(df)}, index=df.index)

        # Concatenate them all together
        df_final = pd.concat([df_left, dots_col, df_right], axis=1)
    else:
        df_final = df

    # Apply the number formatting to the final, potentially truncated DataFrame
    return df_final.map(formatter).to_string()


def image_to_base64(img_obj, fmt="png"):
    """
    Converts a Matplotlib or Plotly figure, or image bytes,
    to a base64 encoded string.
    """
    if isinstance(img_obj, go.Figure): # <-- NEW: Handle Plotly figures
        img_bytes = img_obj.to_image(format=fmt, scale=2)
    elif isinstance(img_obj, matplotlib.figure.Figure):
        buf = io.BytesIO()
        img_obj.savefig(buf, format=fmt, bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.getvalue()
    elif isinstance(img_obj, (bytes, bytearray)):
        img_bytes = img_obj
    else:
        return None
    return base64.b64encode(img_bytes).decode('utf-8')


def generate_adjustment_report_html_pdf(
        final_results,
        template_full_path,
        adjustment_dimension,
        initial_geodetic_params,
        final_geodetic_params,
        values,
        constants,
        hard_constraints=None, soft_constraints=None, vtpv_graph=None,
        chi_graph=None, weight_type=None, adjust_method="Batch Adjustment",
        blunder_test=None, alpha=None, beta=None, initial_results=None,
        network_plot=None, error_ellipse_plots=None, error_ellipse_stats=None
):
    """
    Generates a professional PDF report from geodetic adjustment results.

    This function takes adjustment data, renders it into an HTML template using
    Jinja2, and then converts the final HTML into a PDF file which is returned
    as an in-memory byte buffer. This is ideal for web backends that need to
    serve dynamically generated PDFs for download.

    Args:
        final_results (dict):
            The primary dictionary containing the final adjustment results, such
            as 'X Hat (Final)', 'Residuals', 'Sigma_X_hat_Aposteriori', etc.
        template_full_path (str):
            The absolute file path to the Jinja2 HTML template used for the report.
        adjustment_dimension (str):
            The dimension of the adjustment ('1D', '2D', or '3D'). This controls
            how parameters and ellipses are displayed in the report.
        hard_constraints (any, optional):
            Data representing hard constraints. Defaults to None.
        soft_constraints (any, optional):
            Data representing soft constraints. Defaults to None.
        vtpv_graph (matplotlib.figure.Figure or bytes, optional):
            The VTPV graph object or pre-rendered bytes. Defaults to None.
        chi_graph (matplotlib.figure.Figure or bytes, optional):
            The Chi-Square test graph object or bytes. Defaults to None.
        weight_type (str, optional):
            A string describing the weighting strategy used (e.g., "Standard Deviation").
            Defaults to None.
        adjust_method (str, optional):
            The name of the adjustment method (e.g., "Batch Adjustment").
            Defaults to "Batch Adjustment".
        blunder_test (str, optional):
            The name of the blunder detection test used (e.g., "Baarda's Data Snooping").
            Defaults to None.
        alpha (float, optional):
            The significance level alpha. Defaults to None.
        beta (float, optional):
            The significance level beta for power of the test. Defaults to None.
        initial_results (dict, optional):
            A dictionary of initial parameter values. Defaults to None.
        network_plot (matplotlib.figure.Figure or bytes, optional):
            The network plot graph object or bytes. Defaults to None.
        error_ellipse_plots (list[plotly.graph_objects.Figure], optional):
            A list of Plotly figure objects, each representing an error ellipse
            for a station. The title of each plot must contain the station name.
            Defaults to None.
        error_ellipse_stats (dict, optional):
            A dictionary where keys are station names and values are dicts
            containing ellipse statistics (e.g., {'a': ..., 'b': ..., 'theta_deg': ...}).
            Defaults to None.

    Returns:
        bytes:
            The generated PDF content as a byte buffer. If an error occurs
            during generation, it returns None.
    """
    try:
        # --- 1. Set up Jinja2 Environment ---
        # Locates the template file based on the provided full path.
        template_dir = os.path.dirname(template_full_path)
        template_filename = os.path.basename(template_full_path)
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_filename)

        # --- 2. Prepare and Format Data for the Template ---
        num_obs = len(final_results.get("L Observed", []))
        num_params = len(final_results.get("X Hat (Final)", []))
        # More robust outlier detection check
        outlier_detected = "Yes" if blunder_test and str(blunder_test).lower() != "none" else "No"

        # Flatten arrays for consistent processing and display
        L_obs = np.array(final_results.get("L Observed", []), dtype=float).flatten()
        V = np.array(final_results.get("Residuals", []), dtype=float).flatten()
        L_adj = np.array(final_results.get("L Adjusted", []), dtype=float).flatten()

        Sigma_L = np.array(final_results.get("Sigma_L_Observed", np.zeros((num_obs, num_obs))), dtype=float)
        Sigma_VV = np.array(final_results.get("Sigma_VV", np.zeros((num_obs, num_obs))), dtype=float)
        sigma_Lb = np.sqrt(np.abs(np.diag(Sigma_L)))
        sigma_Vhat = np.sqrt(np.abs(np.diag(Sigma_VV)))
        sigma_Lhat = np.sqrt(np.abs(np.diag(final_results.get("Sigma L Adjusted", np.zeros((num_obs, num_obs))))))

        observation_results = []
        for i in range(num_obs):
            observation_results.append({
                "sno": i + 1, "observed": f"{L_obs[i]:.6f}", "std_dev_obs": f"{sigma_Lb[i]:.6f}",
                "residual": f"{V[i]:.6f}", "adjusted": f"{L_adj[i]:.6f}", "std_dev_adj": f"{sigma_Lhat[i]:.6f}",
            })

        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_residuals = np.nan_to_num(V / sigma_Vhat)
        residuals = []
        for i in range(num_obs):
            residuals.append({
                "sno": i + 1, "residual": f"{V[i]:.6f}",
                "std_dev_residual": f"{sigma_Vhat[i]:.6f}",
                "normalized_residual": f"{abs(normalized_residuals[i]):.4f}",
            })

        # Format observation equations to include their observed value
        equations = final_results.get("Equations", [])
        obs_eq_formatted = [
            f"{eq} = {L_obs[i]:.6f}" if i < len(L_obs) else str(eq)
            for i, eq in enumerate(equations)
        ]
        observation_equations_str = "\n".join(obs_eq_formatted)

        # --- STATION PARAMETER LOGIC (Handles Fixed and Unknown Stations) ---
        param_names = [str(p) for p in final_results.get("PARAMS_Name", [])]
        X_hat_final = np.array(final_results.get("X Hat (Final)", [])).flatten()
        Sigma_X_hat = np.array(
            final_results.get("Sigma_X_hat_Aposteriori", np.zeros((len(param_names), len(param_names)))))
        sigma_Xhat = np.sqrt(np.abs(np.diag(Sigma_X_hat)))

        # Create a map for efficient index lookups to avoid nested loops
        param_to_index = {name: i for i, name in enumerate(param_names)}

        # Get a unique, sorted list of all station names from both unknown and fixed params
        unknown_stations = {p.split('_')[1] for p in param_names if '_' in p}
        fixed_stations = {c.split('_')[1] for c in constants if '_' in c}
        all_stations = sorted(list(unknown_stations.union(fixed_stations)))

        station_map = {}
        for station in all_stations:
            station_map[station] = {
                "name": station,
                "cartesian": {"X": {}, "Y": {}, "Z": {}},
                "geodetic": {"lambda": {}, "phi": {}, "h": {}},
            }

            sym_x, sym_y, sym_z = f"X_{station}", f"Y_{station}", f"Z_{station}"

            # Check if the station is a fixed constant
            if station in fixed_stations:
                # Populate Cartesian data for FIXED stations
                station_map[station]["cartesian"]["X"] = {"initial": f"{constants.get(sym_x, 0):.6f}", "final": "FIXED",
                                                          "sd": "FIXED"}
                station_map[station]["cartesian"]["Y"] = {"initial": f"{constants.get(sym_y, 0):.6f}", "final": "FIXED",
                                                          "sd": "FIXED"}
                station_map[station]["cartesian"]["Z"] = {"initial": f"{constants.get(sym_z, 0):.6f}", "final": "FIXED",
                                                          "sd": "FIXED"}

                # Populate Geodetic data for FIXED stations
                final_geo = final_geodetic_params.get(station, {})
                initial_geo = initial_geodetic_params.get(station, {})
                station_map[station]["geodetic"]["phi"] = {"initial": initial_geo.get("phi", "N/A"), "final": "FIXED",
                                                           "sd": "FIXED"}
                station_map[station]["geodetic"]["lambda"] = {"initial": initial_geo.get("lambda", "N/A"),
                                                              "final": "FIXED", "sd": "FIXED"}
                station_map[station]["geodetic"]["h"] = {"initial": initial_geo.get("h", "N/A"), "final": "FIXED",
                                                         "sd": "FIXED"}

            else:  # The station is an unknown parameter
                # Populate Cartesian data for UNKNOWN stations
                for coord_letter, sym in [("X", sym_x), ("Y", sym_y), ("Z", sym_z)]:
                    if sym in param_to_index:
                        idx = param_to_index[sym]
                        station_map[station]["cartesian"][coord_letter] = {
                            "initial": f"{values.get(sym, 0):.6f}",
                            "final": f"{X_hat_final[idx]:.6f}",
                            "sd": f"{sigma_Xhat[idx]:.6f}"
                        }

                # Populate Geodetic data for UNKNOWN stations
                if station in final_geodetic_params and final_geodetic_params[station]:
                    final_geo = final_geodetic_params[station]
                    initial_geo = initial_geodetic_params.get(station, {})

                    station_map[station]["geodetic"]["phi"] = {
                        "initial": initial_geo.get("phi", "N/A"),
                        "final": final_geo.get("phi", ("N/A",))[0],
                        "sd": final_geo.get("phi", ("", "N/A"))[1]
                    }
                    station_map[station]["geodetic"]["lambda"] = {
                        "initial": initial_geo.get("lambda", "N/A"),
                        "final": final_geo.get("lambda", ("N/A",))[0],
                        "sd": final_geo.get("lambda", ("", "N/A"))[1]
                    }
                    station_map[station]["geodetic"]["h"] = {
                        "initial": initial_geo.get("h", "N/A"),
                        "final": final_geo.get("h", ("N/A",))[0],
                        "sd": final_geo.get("h", ("", "N/A"))[1]
                    }

        stations = list(station_map.values())

        # Process error ellipse plots provided as input
        error_ellipses_data = []
        if error_ellipse_plots and error_ellipse_stats:
            for station_name, stats in error_ellipse_stats.items():
                plot_for_station = next(
                    (p for p in error_ellipse_plots if station_name in p.layout.title.text),
                    None
                )
                if plot_for_station:
                    img_bytes = plot_for_station.to_image(format="png", scale=2)
                    b64_img = image_to_base64(img_bytes)
                    error_ellipses_data.append({
                        "name": station_name,
                        "stats": stats,
                        "image": b64_img
                    })

        # --- Prepare Formatted Strings for Constraints ---
        constraints_parts, used_constraint_types = [], []
        if hard_constraints:
            used_constraint_types.append("Hard")
            constraints_parts.append("\n".join(
                ["Hard Constraints:"] + [f"  • Station {st}: Fixed at {val}" for st, val in hard_constraints.items()]
            ))
        if soft_constraints:
            used_constraint_types.append("Soft")
            sc_lines = ["Soft Constraints:"]
            for station, data in soft_constraints.items():
                sc_lines.append(f"  • Station {station}: Constrained at {data['value']}")
                sc_lines.append(f"    VCV Matrix:\n{np.array2string(np.array(data['cov']), prefix='    ')}")
            constraints_parts.append("\n".join(sc_lines))
        constraints_display_str = "\n\n".join(constraints_parts) if constraints_parts else "None"
        constraints_used_str = ", ".join(used_constraint_types) if used_constraint_types else "None"

        # Safely extract the A Posteriori Variance as a scalar
        aposteriori_var_raw = final_results.get('Aposteriori Variance', 0)
        aposteriori_var_scalar = np.array(aposteriori_var_raw).flatten()[0]

        # --- 3. Build the Master Context Dictionary ---
        context = {
            # Report Metadata
            "curr_date": datetime.today().strftime("%Y-%m-%d"),
            "curr_time": datetime.today().strftime("%H:%M:%S"),

            # Executive Summary Data
            "adjustment_dimension": adjustment_dimension,  # **NEWLY ADDED**
            "num_observations": num_obs,
            "num_params": num_params,
            "dof": final_results.get("DOF", "N/A"),
            "apriori_var": final_results.get('Apriori Variance', 0),
            "aposteriori_var": f"{aposteriori_var_scalar:.6f}",
            "outlier_detected": outlier_detected,
            "blunder_test": blunder_test if outlier_detected == "Yes" else "N/A",
            "weight_type": weight_type,
            "constraints_used": constraints_used_str,
            "adjust_method": adjust_method,
            "alpha": alpha,
            "beta": beta,

            # Formatted Data Sections
            "observation_equations": observation_equations_str,
            "constraints": constraints_display_str,

            # Data for Tables
            "observation_results": observation_results,
            "residuals": residuals,
            "stations": stations,

            # Covariance Matrices (converted to string for pre-formatted display)
            "covar_L_adj": format_matrix_for_report(final_results.get("Sigma L Adjusted", [])),
            "covar_param": format_matrix_for_report(final_results.get("Sigma_X_hat_Aposteriori", [])),
            "covar_residual": format_matrix_for_report(final_results.get("Sigma_VV", [])),

            # Base64 Encoded Graphs
            "vtpv_graph": image_to_base64(vtpv_graph),
            "chi_graph": image_to_base64(chi_graph),
            "network_plot": image_to_base64(network_plot),
            "error_ellipses": error_ellipses_data
        }

        # --- 4. Render HTML and Generate PDF Buffer ---
        html_out = template.render(context)
        pdf_buffer = HTML(string=html_out).write_pdf()

        print("✅ Successfully generated PDF in memory buffer.")
        return pdf_buffer

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None