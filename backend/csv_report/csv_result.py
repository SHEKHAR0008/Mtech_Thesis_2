import numpy as np
import pandas as pd
from io import BytesIO
import math


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
                                          If None, highlighting is based on final_results.

    Returns:
        tuple: A tuple containing two BytesIO objects: (obs_buffer, param_buffer)
    """
    report_source = initial_results if initial_results is not None else final_results

    # --- Extract data safely from the source dictionary ---
    Labels = report_source["Labels"]
    L_obs = np.array(report_source.get("L Observed", []), dtype=float).flatten()
    V = np.array(report_source.get("Residuals", []), dtype=float).flatten()
    L_adj = np.array(report_source.get("L Adjusted", []), dtype=float).flatten()
    sigma_VV = np.array(report_source.get("Sigma_VV", np.zeros((len(V), len(V)))), dtype=float)
    sigma_L_adj = np.array(report_source.get("Sigma L Adjusted", np.zeros((len(L_adj), len(L_adj)))), dtype=float)

    sigma_Lb = np.sqrt(np.abs(np.diag(sigma_VV))) if sigma_VV.size else [np.nan] * len(L_obs)
    sigma_Vhat = np.sqrt(np.abs(np.diag(sigma_VV))) if sigma_VV.size else [np.nan] * len(V)
    sigma_Lhat = np.sqrt(np.abs(np.diag(sigma_L_adj))) if sigma_L_adj.size else [np.nan] * len(L_adj)
    normalized_residuals = (V / sigma_Vhat) if np.all(sigma_Vhat > 0) else [0] * len(V)

    # --- Observation DataFrame (with numerical data) ---
    obs_data = {
        "Sr.": [i + 1 for i in range(len(Labels))],
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

    # --- Parameters DataFrame (based on FINAL results) ---
    Params_name = [str(p) for p in final_results.get("PARAMS_Name", [])]
    X_hat = np.array(final_results.get("X Hat (Final)", []), dtype=float).flatten()
    Sigma_X = np.array(final_results.get("Sigma_X_hat_Aposteriori", np.zeros((len(X_hat), len(X_hat)))), dtype=float)
    sigma_X = np.sqrt(np.abs(np.diag(Sigma_X))) if Sigma_X.size else [np.nan] * len(X_hat)

    param_data = {"Parameter": Params_name, "Estimate": X_hat, "SD": sigma_X}
    if geodetic_coords is not None:
        param_data["Geodetic Coord"] = [gc[0] for gc in geodetic_coords]
        param_data["Geodetic SD"] = [gc[1] for gc in geodetic_coords]
    df_params = pd.DataFrame(param_data)

    constants = final_results.get("Constant", {})
    const_rows = pd.DataFrame([{"Parameter": str(k), "Estimate": v, "SD": "fixed"} for k, v in constants.items()])
    df_all_params = pd.concat([const_rows, df_params], ignore_index=True)

    # --- Prepare for writing to Excel buffers ---
    obs_buffer, param_buffer = BytesIO(), BytesIO()
    removed_indices = {res['removed_index'] for res in outlier_results} if outlier_results else set()

    # --- Write Observations to Excel with Formatting ---
    with pd.ExcelWriter(obs_buffer, engine="xlsxwriter") as writer:
        df_obs.to_excel(writer, index=False, sheet_name="Observations")
        workbook = writer.book
        worksheet = writer.sheets["Observations"]

        # Define formats
        red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        yellow_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        bold_format = workbook.add_format({'bold': True})
        header_format = workbook.add_format({'bold': True, 'font_size': 12, 'underline': True})

        # Number formats for controlled precision
        num_format_6dp = workbook.add_format({"num_format": "0.000000"})
        num_format_3dp = workbook.add_format({"num_format": "0.000"})
        text_format = workbook.add_format({"num_format": "@"})  # For labels

        # Apply column formats
        worksheet.set_column('A:A', 5, text_format)  # Sr.
        worksheet.set_column('B:B', 20, text_format)  # Label
        worksheet.set_column('C:H', 22, num_format_6dp)  # Numerical data
        worksheet.set_column('I:I', 25, num_format_3dp)  # Test Statistic

        # Apply conditional formatting for outlier highlighting
        for row_num in range(len(df_obs)):
            if abs(normalized_residuals[row_num]) > rejection_level:
                worksheet.set_row(row_num + 1, cell_format=yellow_format)
            if row_num in removed_indices:
                worksheet.set_row(row_num + 1, cell_format=red_format)

        # Add outlier summary
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

    # --- Write Parameters to Excel with Formatting ---
    with pd.ExcelWriter(param_buffer, engine="xlsxwriter") as writer:
        df_all_params.to_excel(writer, index=False, sheet_name="Parameters")
        workbook = writer.book
        worksheet = writer.sheets["Parameters"]

        worksheet.set_column('A:A', 25, text_format)  # Parameter Name/Constant Name
        worksheet.set_column('B:E', 25, num_format_6dp)  # Estimate, SD, Geodetic...

        # Add the same outlier summary
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

