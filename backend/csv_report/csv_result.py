import numpy as np
import pandas as pd
from io import BytesIO

def export_adjustment_results_excel(
        final_results,
        outlier_results,
        blunder_detection_method,
        alpha,
        beta_power,
        initial_geodetic_params,
        final_geodetic_params,
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
    sigma_L = np.array(report_source.get("Sigma_L_Observed", np.zeros((len(L_obs), len(L_obs)))), dtype=float)

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

    # Initialize columns with a flexible 'object' dtype instead of a numeric one.
    df_all_params["Geodetic Coord"] = pd.Series(dtype='object')
    df_all_params["Geodetic SD"] = pd.Series(dtype='object')

    # Get unique station names from the parameters
    stations = sorted(list(set(p.split('_')[1] for p in Params_name if '_' in p)))

    for station in stations:
        # Check if geodetic data exists for the station
        if station in final_geodetic_params and final_geodetic_params[station]:
            final_geo = final_geodetic_params[station]

            # Create a clean, readable string for the coordinates
            geodetic_string = {"X":f"Lat: {final_geo['phi'][0]}",
                               "Y":f"Lon: {final_geo['lambda'][0]}",
                               "Z":f"Hgt: {final_geo['h'][0]}"}

            # Create a clean, readable string for the standard deviations
            sd_string = {"X":f"dLat: {final_geo['phi'][1]}",
                         "Y":f"dLon: {final_geo['lambda'][1]}",
                         "Z":f"dHgt: {final_geo['h'][1]}"}

            # Assign the strings to the correct rows in the DataFrame
            for coord in ['X', 'Y', 'Z']:
                param_name = f"{coord}_{station}"
                # Use .isin() for robust checking
                if df_all_params["Parameter"].isin([param_name]).any():
                    df_all_params.loc[df_all_params["Parameter"] == param_name, "Geodetic Coord"] = geodetic_string[coord]
                    df_all_params.loc[df_all_params["Parameter"] == param_name, "Geodetic SD"] = sd_string[coord]

    # Fill any remaining empty cells in these columns to avoid issues
    df_all_params["Geodetic Coord"].fillna("", inplace=True)
    df_all_params["Geodetic SD"].fillna("", inplace=True)
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
        worksheet.write(last_row + 2, 0, "Significance Level (α):", bold_format)
        worksheet.write(last_row + 2, 1, alpha)
        worksheet.write(last_row + 3, 0, "Power of Test (1-β):", bold_format)
        worksheet.write(last_row + 3, 1, beta_power)
        if blunder_detection_method != "None":
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
        worksheet.write(last_row + 2, 0, "Significance Level (α):", bold_format)
        worksheet.write(last_row + 2, 1, alpha)
        worksheet.write(last_row + 3, 0, "Power of Test (1-β):", bold_format)
        worksheet.write(last_row + 3, 1, beta_power)
        if blunder_detection_method != "None":
            worksheet.write(last_row + 4, 0, "Total Outliers Detected:", bold_format)
            worksheet.write(last_row + 4, 1, len(removed_indices))

        # --- 4. NEW: Create Combined Variance-Covariance Excel Buffer ---
        covar_buffer = BytesIO()

        # Create DataFrames for each matrix with appropriate labels for clarity
        df_cov_L_adj = pd.DataFrame(final_results.get("Sigma L Adjusted", []), index=Labels, columns=Labels)
        df_cov_V = pd.DataFrame(final_results.get("Sigma_VV", []), index=Labels, columns=Labels)
        df_cov_X = pd.DataFrame(final_results.get("Sigma_X_hat_Aposteriori", []), index=Params_name,
                                columns=Params_name)

        # Use ExcelWriter to save each DataFrame to a different sheet in the same file
        with pd.ExcelWriter(covar_buffer, engine="xlsxwriter") as writer:
            df_cov_L_adj.to_excel(writer, index=True, sheet_name="Covar_Adjusted_Obs")
            df_cov_V.to_excel(writer, index=True, sheet_name="Covar_Residuals")
            df_cov_X.to_excel(writer, index=True, sheet_name="Covar_Parameters")

            # Optional: Add some basic formatting to auto-fit column widths for readability
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df_cov_L_adj):  # Get column names
                    series = df_cov_L_adj[col]
                    max_len = max((
                        series.astype(str).map(len).max(),  # len of largest item
                        len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                    worksheet.set_column(idx, idx, max_len)  # set column width

    obs_buffer.seek(0)
    param_buffer.seek(0)
    return obs_buffer, param_buffer, covar_buffer