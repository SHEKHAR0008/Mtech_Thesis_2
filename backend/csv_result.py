import math
import numpy as np
import pandas as pd
from io import BytesIO

import math

def format_number(num, sig=3):
    """
    Custom formatter:
    - Expand any scientific-notation number first to its full float value
    - If abs(value) < 1: use normalized scientific notation with 3 decimals
    - If abs(value) >= 1: show full value with 3 decimals, attach e+00
    """
    try:
        if num is None:
            return ""
        num = float(num)  # ensures expansion from sci notation
        if math.isnan(num) or math.isinf(num):
            return ""
        if num == 0:
            return "0.000e+00"

        absnum = abs(num)

        if absnum < 1:  # small numbers → scientific
            exp = int(math.floor(math.log10(absnum)))
            mantissa = num / (10**exp)
            return f"{mantissa:.3f}e{exp:+02d}"
        else:  # large numbers → full value, 3 decimals, force e+00
            return f"{num:.3f}e+00"

    except Exception:
        return str(num)


def export_adjustment_results_csv(final_results, rejection_level=3.0, geodetic_coords=None):
    """
    Returns two Excel files as bytes:
    1. Observations.xlsx (with residuals, variances, etc.)
    2. Parameters.xlsx (with SD, optional geodetic coords)
    All numbers are formatted as strings to prevent Excel auto-formatting.
    """

    # --- Extract safely ---
    Labels = final_results["Labels"]
    L_obs = np.array(final_results.get("L Observed", []), dtype=float).flatten()
    V = np.array(final_results.get("Residuals", []), dtype=float).flatten()
    L_adj = np.array(final_results.get("L Adjusted", []), dtype=float).flatten()

    sigma_VV = np.array(final_results.get("Sigma_VV", np.zeros((len(V), len(V)))), dtype=float)
    sigma_L_adj = np.array(final_results.get("Sigma L Adjusted", np.zeros((len(L_adj), len(L_adj)))), dtype=float)

    sigma_Lb = np.sqrt(np.abs(np.diag(sigma_VV))) if sigma_VV.size else [None] * len(L_obs)
    sigma_Vhat = np.sqrt(np.abs(np.diag(sigma_VV))) if sigma_VV.size else [None] * len(V)
    sigma_Lhat = np.sqrt(np.abs(np.diag(sigma_L_adj))) if sigma_L_adj.size else [None] * len(L_adj)

    # Apply formatting (all to strings)
    L_obs = [format_number(x) for x in L_obs]
    L_adj = [format_number(x) for x in L_adj]
    sigma_Lb = [format_number(x) for x in sigma_Lb]
    sigma_Lhat = [format_number(x) for x in sigma_Lhat]
    V_str = [format_number(x) for x in V]
    sigma_Vhat_str = [format_number(x) for x in sigma_Vhat]

    # Normalized residuals
    norm_res = []
    for vi, svi in zip(V, sigma_Vhat):
        if svi is None or svi == 0:
            norm_res.append("")
        else:
            norm_res.append(f"{vi/svi:.3f}")

    # --- Observation DataFrame ---
    obs_data = []
    for i, (label, lo, so, v, sv, la, sla, nr) in enumerate(
            zip(Labels, L_obs, sigma_Lb, V_str, sigma_Vhat_str, L_adj, sigma_Lhat, norm_res), start=1):
        obs_data.append({
            "Sr.": str(i),
            "Label": str(label),
            "L_Observed": lo,
            "Std_Dev_L_Observed": so,
            "Residual": v,
            "Std_Dev_Residual": sv,
            "L_adjusted": la,
            "Std_Dev_L_adjusted": sla,
            "rᵢ (%)": "",
            f"Normalized residual (rej={rejection_level})": nr,
            "MDB": "",
            "Expected model error": ""
        })

    df_obs = pd.DataFrame(obs_data, dtype=str)

    # --- Parameters DataFrame ---
    Params_name = final_results.get("PARAMS_Name", [])
    X_hat = np.array(final_results.get("X Hat (Final)", []), dtype=float).flatten()
    Sigma_X = np.array(final_results.get("Sigma_X_hat_Aposteriori", np.zeros((len(X_hat), len(X_hat)))), dtype=float)
    sigma_X = np.sqrt(np.abs(np.diag(Sigma_X))) if Sigma_X.size else [None] * len(X_hat)

    param_data = []
    for i, (x, sx) in enumerate(zip(X_hat, sigma_X)):
        entry = {
            "Parameter": Params_name[i] if isinstance(Params_name, (list, tuple, pd.Series)) else f"Param_{i+1}",
            "Estimate": format_number(x),
            "SD": format_number(sx),
        }
        if geodetic_coords is not None and i < len(geodetic_coords):
            entry["Geodetic Coord"] = format_number(geodetic_coords[i][0])
            entry["Geodetic SD"] = (
                format_number(geodetic_coords[i][1]) if len(geodetic_coords[i]) > 1 else ""
            )
        else:
            entry["Geodetic Coord"] = ""
            entry["Geodetic SD"] = ""
        param_data.append(entry)

    df_params = pd.DataFrame(param_data, dtype=str)

    # Constants to DataFrame
    constants = final_results.get("Constant", {})
    const_rows = pd.DataFrame([
        {
            "Parameter": str(k),
            "Estimate": format_number(v),
            "SD": "fixed",
            "Geodetic Coord": "",
            "Geodetic SD": ""
        }
        for k, v in constants.items()
    ], dtype=str)

    # Combine constants + params
    df_all = pd.concat([const_rows, df_params], ignore_index=True)

    # --- Save Observations to Excel ---
    obs_buffer = BytesIO()
    with pd.ExcelWriter(obs_buffer, engine="xlsxwriter") as writer:
        df_obs.to_excel(writer, index=False, sheet_name="Observations")
        workbook = writer.book
        text_format = workbook.add_format({"num_format": "@", "font_name": "Courier New"})
        worksheet = writer.sheets["Observations"]
        worksheet.set_column(0, len(df_obs.columns), 20, text_format)
    obs_buffer.seek(0)

    # --- Save Parameters to Excel ---
    param_buffer = BytesIO()
    with pd.ExcelWriter(param_buffer, engine="xlsxwriter") as writer:
        df_all.to_excel(writer, index=False, sheet_name="Parameters")
        workbook = writer.book
        text_format = workbook.add_format({"num_format": "@", "font_name": "Courier New"})
        worksheet = writer.sheets["Parameters"]
        worksheet.set_column(0, len(df_all.columns), 20, text_format)
    param_buffer.seek(0)

    return obs_buffer, param_buffer
