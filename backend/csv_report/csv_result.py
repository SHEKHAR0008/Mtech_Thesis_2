import numpy as np
import pandas as pd
from io import BytesIO

def export_adjustment_results_excel(
        final_results,
        outlier_results,
        blunder_detection_method,
        alpha,
        beta_power,
        rejection_level=3.0,
        geodetic_coords=None,
        initial_results=None
):
    """
    Returns two Excel files as bytes with detailed adjustment and outlier detection results.
    Values are stored as numerical types with controlled display precision.

    Args:
        final_results (dict): The dictionary of adjustment results after all processing.
        outlier_results (list): A list of dictionaries, where each entry represents an iteration
                                of outlier removal. Can be empty if no outliers were found/run.
        blunder_detection_method (str): The name of the outlier detection method used.
        alpha (float): The significance level (Type I error probability).
        beta_power (float): The desired power of the test (1 - beta).
        rejection_level (float): The critical value for flagging outliers.
        geodetic_coords (list, optional): Geodetic coordinates for the parameters.
        initial_results (dict, optional): The results of the *very first* adjustment,
                                          before any outliers were removed. This is crucial
                                          for correctly highlighting all potential outliers.

    Returns:
        tuple: A tuple containing two BytesIO objects: (obs_buffer, param_buffer)
    """
    report_source = initial_results if initial_results is not None else final_results

    # --- 1. Extract and Prepare Observation Data ---
    Labels = report_source.get("Labels", [])
    L_obs = np.array(report_source.get("L Observed", []), dtype=float).flatten()
    V = np.array(report_source.get("Residuals", []), dtype=float).flatten()
    L_adj = np.array(report_source.get("L Adjusted", []), dtype=float).flatten()
    sigma_VV = np.array(report_source.get("Sigma_VV", np.zeros((len(V), len(V)))), dtype=float)
    sigma_L_adj = np.array(report_source.get("Sigma L Adjusted", np.zeros((len(L_adj), len(L_adj)))), dtype=float)
    sigma_L = np.array(report_source.get("Sigma_L", np.zeros((len(L_obs), len(L_obs)))), dtype=float)

    # Use np.abs() to prevent sqrt of small negative numbers from numerical instability
    sigma_Lb = np.sqrt(np.abs(np.diag(sigma_L)))
    sigma_Vhat = np.sqrt(np.abs(np.diag(sigma_VV)))
    sigma_Lhat = np.sqrt(np.abs(np.diag(sigma_L_adj)))

    # Prevent division by zero if a standard deviation is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_residuals = np.nan_to_num(V / sigma_Vhat)

    obs_data = {
        "Sr.": range(1, len(Labels) + 1),
        "Label": Labels,
        "L_Observed": L_obs,
        "Std_Dev_L_Observed": sigma_Lb,
        "Residual": V,
        "Std_Dev_Residual": sigma_Vhat,
        "L_adjusted": L_adj,
        "Std_Dev_L_adjusted": sigma_Lhat,
        f"Test Statistic (|wᵢ| > {rejection_level:.3f})": np.abs(normalized_residuals),
    }
    df_obs = pd.DataFrame(obs_data)

    # --- 2. Extract and Prepare Parameter Data ---
    Params_name = [str(p) for p in final_results.get("PARAMS_Name", [])]
    X_hat = np.array(final_results.get("X Hat (Final)", []), dtype=float).flatten()
    Sigma_X = np.array(final_results.get("Sigma_X_hat_Aposteriori", np.zeros((len(X_hat), len(X_hat)))), dtype=float)
    sigma_X = np.sqrt(np.abs(np.diag(Sigma_X)))

    # Create DataFrame for unknown parameters
    param_data = {"Parameter": Params_name, "Estimate": X_hat, "SD": sigma_X}
    df_params = pd.DataFrame(param_data)

    # Create DataFrame for fixed constants and combine them
    constants = final_results.get("Constant", {})
    if constants:
        const_rows = pd.DataFrame([
            {"Parameter": str(k), "Estimate": v, "SD": "fixed"} for k, v in constants.items()
        ])
        df_all_params = pd.concat([const_rows, df_params], ignore_index=True)
    else:
        df_all_params = df_params

    # Add geodetic coordinate columns, ensuring they exist even if empty

    df_all_params["Geodetic Coord"] = np.nan
    df_all_params["Geodetic SD"] = np.nan

    if geodetic_coords:  # Only populate if the list is not empty
        if len(geodetic_coords) == len(Params_name):
            try:
                coord_map = {name: gc[0] for name, gc in zip(Params_name, geodetic_coords)}
                sd_map = {name: gc[1] for name, gc in zip(Params_name, geodetic_coords)}
                df_all_params["Geodetic Coord"] = df_all_params["Parameter"].map(coord_map)
                df_all_params["Geodetic SD"] = df_all_params["Parameter"].map(sd_map)
            except (IndexError, TypeError):
                print("Warning: Geodetic coordinates list is malformed. Columns will remain empty.")
        else:
            print(f"Warning: Mismatch in length of parameters and geodetic coordinates. Columns will remain empty.")

    # --- 3. Write DataFrames to In-Memory Excel Files ---
    obs_buffer, param_buffer = BytesIO(), BytesIO()
    removed_indices = {res['removed_index'] for res in outlier_results} if outlier_results else set()

    # --- Write Observations Excel File ---
    with pd.ExcelWriter(obs_buffer, engine="xlsxwriter") as writer:
        df_obs.to_excel(writer, index=False, sheet_name="Observations")
        workbook = writer.book
        worksheet = writer.sheets["Observations"]
        red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        yellow_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        bold_format = workbook.add_format({'bold': True})
        header_format = workbook.add_format({'bold': True, 'font_size': 12, 'underline': True})
        num_format_6dp = workbook.add_format({"num_format": "0.000000"})
        num_format_3dp = workbook.add_format({"num_format": "0.000"})
        text_format = workbook.add_format({"num_format": "@"})

        worksheet.set_column('A:A', 5, text_format)
        worksheet.set_column('B:B', 20, text_format)
        worksheet.set_column('C:H', 22, num_format_6dp)
        worksheet.set_column('I:I', 25, num_format_3dp)

        for row_num in range(len(df_obs)):
            if abs(normalized_residuals[row_num]) > rejection_level:
                worksheet.set_row(row_num + 1, cell_format=yellow_format)
            if row_num in removed_indices:
                worksheet.set_row(row_num + 1, cell_format=red_format)

        last_row = len(df_obs) + 3
        worksheet.write(last_row, 0, "Outlier Detection Summary", header_format)
        worksheet.write(last_row + 1, 0, "Detection Method:", bold_format)
        worksheet.write(last_row + 1, 1, blunder_detection_method)
        if blunder_detection_method != "None":
            worksheet.write(last_row + 2, 0, "Significance Level (α):", bold_format)
            worksheet.write(last_row + 2, 1, alpha)
            worksheet.write(last_row + 3, 0, "Power of Test (1-β):", bold_format)
            worksheet.write(last_row + 3, 1, beta_power)
            worksheet.write(last_row + 4, 0, "Total Outliers Detected:", bold_format)
            worksheet.write(last_row + 4, 1, len(removed_indices))
            if outlier_results:
                worksheet.write(last_row + 6, 0, "Removal Sequence:", bold_format)
                for i, res in enumerate(outlier_results):
                    original_idx = res['removed_index']
                    label = report_source["Labels"][original_idx]
                    worksheet.write(last_row + 7 + i, 0, f"  Iteration {i + 1}:", bold_format)
                    worksheet.write(last_row + 7 + i, 1, f"Removed '{label}' (Original Index: {original_idx})")

    # --- Write Parameters Excel File ---
    with pd.ExcelWriter(param_buffer, engine="xlsxwriter") as writer:
        df_all_params.fillna('').to_excel(writer, index=False, sheet_name="Parameters")
        workbook = writer.book
        worksheet = writer.sheets["Parameters"]

        num_format_6dp = workbook.add_format({"num_format": "0.000000"})
        text_format = workbook.add_format({"num_format": "@"})
        header_format = workbook.add_format({'bold': True, 'font_size': 12, 'underline': True})
        bold_format = workbook.add_format({'bold': True})

        worksheet.set_column('A:A', 25, text_format)
        worksheet.set_column('B:B', 25, num_format_6dp)
        worksheet.set_column('C:C', 25, num_format_6dp)
        worksheet.set_column('D:E', 25, num_format_6dp)

        last_row = len(df_all_params) + 3
        worksheet.write(last_row, 0, "Outlier Detection Summary", header_format)
        worksheet.write(last_row + 1, 0, "Detection Method:", bold_format)
        worksheet.write(last_row + 1, 1, blunder_detection_method)
        if blunder_detection_method != "None":
            worksheet.write(last_row + 2, 0, "Significance Level (α):", bold_format)
            worksheet.write(last_row + 2, 1, alpha)
            worksheet.write(last_row + 3, 0, "Power of Test (1-β):", bold_format)
            worksheet.write(last_row + 3, 1, beta_power)
            worksheet.write(last_row + 4, 0, "Total Outliers Detected:", bold_format)
            worksheet.write(last_row + 4, 1, len(removed_indices))

    obs_buffer.seek(0)
    param_buffer.seek(0)
    return obs_buffer, param_buffer