import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import matplotlib.figure
import os

def image_to_base64(img_obj, fmt="png"):
    """Converts a Matplotlib figure or image bytes to a base64 encoded string."""
    if isinstance(img_obj, matplotlib.figure.Figure):
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
        outlier_detected = "Yes" if blunder_test and blunder_test != "None" else "No"
        
        # Flatten arrays for consistent processing and display
        L_obs = np.array(final_results.get("L Observed", []), dtype=float).flatten()
        V = np.array(final_results.get("Residuals", []), dtype=float).flatten()
        L_adj = np.array(final_results.get("L Adjusted", []), dtype=float).flatten()

        Sigma_L = np.array(final_results.get("Sigma_L", np.zeros((num_obs, num_obs))), dtype=float)
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
                "std_dev_residual": f"{sigma_Vhat[i]:.6f}", "normalized_residual": f"{abs(normalized_residuals[i]):.4f}",
            })

        # Format observation equations to include their observed value
        equations = final_results.get("Equations", [])
        obs_eq_formatted = [
            f"{eq} = {L_obs[i]:.6f}" if i < len(L_obs) else str(eq)
            for i, eq in enumerate(equations)
        ]
        observation_equations_str = "\n".join(obs_eq_formatted)
        
        
        # --- FULL STATION LOGIC ---
        param_names = [str(p) for p in final_results.get("PARAMS_Name", [])]
        X_hat_final = np.array(final_results.get("X Hat (Final)", [])).flatten()
        Sigma_X_hat = np.array(final_results.get("Sigma_X_hat_Aposteriori", np.zeros((num_params, num_params))))
        sigma_Xhat = np.sqrt(np.abs(np.diag(Sigma_X_hat)))
        station_map = {}
        for i, param_name in enumerate(param_names):
            coord, station_name = param_name.split("_", 1) if "_" in param_name else (param_name, "Unknown")
            if station_name not in station_map:
                station_map[station_name] = {
                    "name": station_name,
                    "cartesian": {"X": {}, "Y": {}, "Z": {}},
                    "geodetic": {"lambda": {}, "phi": {}, "h": {}},
                }

            initial_val_raw = initial_results.get(param_name, "") if initial_results else ""
            initial_val_str = ""
            if isinstance(initial_val_raw, (int, float)):
                initial_val_str = f"{initial_val_raw:.6f}"
            elif isinstance(initial_val_raw, (list, np.ndarray)) and len(initial_val_raw) > 0:
                initial_val_str = f"{initial_val_raw[0]:.6f}"

            param_data = {
                "initial": initial_val_str,
                "final": f"{X_hat_final[i]:.6f}",
                "sd": f"{sigma_Xhat[i]:.6f}"
            }

            coord_lower = coord.lower()
            if coord_lower in ["x", "y", "z"]:
                station_map[station_name]["cartesian"][coord.upper()] = param_data
            elif coord_lower in ["lambda", "phi", "h", "height"]:
                key = "h" if coord_lower == "height" else coord_lower
                station_map[station_name]["geodetic"][key] = param_data
        stations = list(station_map.values())

        # Process error ellipse plots provided as input
        error_ellipses_data = []
        if error_ellipse_plots and error_ellipse_stats:
            for station_name, stats in error_ellipse_stats.items():
                # Find the correct Plotly figure by matching the station name in its title
                plot_for_station = next(
                    (p for p in error_ellipse_plots if station_name in p.layout.title.text),
                    None
                )
                if plot_for_station:
                    # Convert the Plotly figure to a PNG image in memory
                    img_bytes = plot_for_station.to_image(format="png", scale=2) # Higher scale for better PDF quality
                    # Encode the image bytes into a base64 string for embedding in HTML
                    b64_img = image_to_base64(img_bytes)
                    error_ellipses_data.append({
                        "name": station_name,
                        "stats": stats,
                        "image": b64_img
                    })

        # --- Prepare Formatted Strings for Constraints ---
        constraints_parts = []
        used_constraint_types = []

        # Process and format hard constraints if they exist
        if hard_constraints:
            used_constraint_types.append("Hard")
            hc_lines = ["Hard Constraints:"]
            for station, value in hard_constraints.items():
                hc_lines.append(f"  • Station {station}: Fixed at {value}")
            constraints_parts.append("\n".join(hc_lines))

        # Process and format soft constraints if they exist
        if soft_constraints:
            used_constraint_types.append("Soft")
            sc_lines = ["Soft Constraints:"]
            for station, data in soft_constraints.items():
                sc_lines.append(f"  • Station {station}: Constrained at {data['value']}")
                # Convert the VCV list to a formatted numpy array string
                cov_matrix_str = np.array2string(np.array(data['cov']), prefix='    ')
                sc_lines.append(f"    VCV Matrix:\n{cov_matrix_str}")
            constraints_parts.append("\n".join(sc_lines))

        # Create the final string for the detailed constraints section
        constraints_display_str = "\n\n".join(constraints_parts) if constraints_parts else "None"

        # Create the final summary string for the executive summary table
        constraints_used_str = ", ".join(used_constraint_types) if used_constraint_types else "None"

        # --- Safely extract the A Posteriori Variance as a scalar ---
        # Get the raw value, which might be a scalar or a single-element array
        aposteriori_var_raw = final_results.get('Aposteriori Variance', 0)

        # Convert to a NumPy array, flatten it, and get the first (and only) element
        aposteriori_var_scalar = np.array(aposteriori_var_raw).flatten()[0]
        # --- 3. Build the Master Context Dictionary ---
        # This dictionary holds all the data that will be passed to the HTML template.
        context = {
            # Report Metadata
            "curr_date": datetime.today().strftime("%Y-%m-%d"),
            "curr_time": datetime.today().strftime("%H:%M:%S"),

            # Executive Summary Data
            "num_observations": num_obs,
            "num_params": num_params,
            "dof": final_results.get("DOF", "N/A"),
            "apriori_var": final_results.get('Apriori Variance', 0),
            "aposteriori_var": f"{aposteriori_var_scalar:.6f}",
            "outlier_detected": outlier_detected,
            "blunder_test": blunder_test if outlier_detected == "Yes" else "N/A",
            "weight_type": weight_type,
            "constraints_used":  constraints_used_str,
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
            "covar_L_adj": pd.DataFrame(final_results.get("Sigma L Adjusted", [])).to_string(),
            "covar_param": pd.DataFrame(final_results.get("Sigma_X_hat_Aposteriori", [])).to_string(),
            "covar_residual": pd.DataFrame(final_results.get("Sigma_VV", [])).to_string(),

            # Base64 Encoded Graphs
            "vtpv_graph": image_to_base64(vtpv_graph),
            "chi_graph": image_to_base64(chi_graph),
            "network_plot": image_to_base64(network_plot),
            "error_ellipses": error_ellipses_data
        }

        # --- 4. Render HTML and Generate PDF Buffer ---
        # Render the Jinja2 template with all the prepared data.
        html_out = template.render(context)

        # Use WeasyPrint to convert the rendered HTML string into a PDF in memory.
        pdf_buffer = HTML(string=html_out).write_pdf()

        print("✅ Successfully generated PDF in memory buffer.")
        return pdf_buffer

    except Exception as e:
        # If any error occurs during the process, print it and return None.
        print(f"❌ Error generating report: {e}")
        return None