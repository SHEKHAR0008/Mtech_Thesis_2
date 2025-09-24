from docxtpl import DocxTemplate, InlineImage
from docx.shared import Inches
import io, math
from datetime import datetime
import os
import matplotlib.figure

def format_number(num, sig=3):
    """Format numbers into scientific or fixed notation."""
    try:
        num = float(num)
        if math.isnan(num) or math.isinf(num):
            return ""
        if num == 0:
            return "0.000"
        if (abs(num) < 1e-3) or (abs(num) > 1e5):
            return f"{num:.{sig}e}"
        else:
            return f"{num:.3f}"
    except Exception:
        return str(num)

def _to_image_bytes(img_obj, fmt="png"):
    """
    Convert input to image bytes:
    - matplotlib Figure -> PNG bytes
    - bytes / BytesIO   -> return as-is
    - file path (str)   -> read bytes
    """
    # Case 1: Matplotlib Figure
    try:

        if isinstance(img_obj, matplotlib.figure.Figure):
            buf = io.BytesIO()
            img_obj.savefig(buf, format=fmt, bbox_inches="tight")
            buf.seek(0)
            return buf.getvalue()
    except ImportError:
        pass

    # Case 2: BytesIO
    if isinstance(img_obj, io.BytesIO):
        return img_obj.getvalue()

    # Case 3: Raw bytes
    if isinstance(img_obj, (bytes, bytearray)):
        return img_obj

    # Case 4: File path
    if isinstance(img_obj, str) and os.path.exists(img_obj):
        with open(img_obj, "rb") as f:
            return f.read()

    return None

def generate_adjustment_report_docx_pdf(
    final_results,
    template_path,
    hard_constraints=None,
    soft_constraints=None,
    vtpv_graph=None,
    chi_graph=None,
    weight_type="Unity/Full/Diagonal",
    error_ellipse = None ,
    network_plot = None ,
    outlier_result = None ,
    blunder_detection_method = None ,
    alpha = None ,
    beta_power = None ,
    rejection_level = None ,
    geodetic_coords = None ,
    initial_results = None ,
):
    """
    Generate Adjustment Report (DOCX buffer).

    Fills placeholders:
      {{num_observations}}, {{num_params}}, {{dof}},
      {{apriori_var}}, {{aposteriori_var}},
      {{outlier_detected}}, {{weight_type}}, {{constraints_used}},
      {{observation_equations}}, {{constraints}},
      {{vtpv_graph}}, {{chi_graph}},
      {{curr_date}}, {{curr_time}}
    """

    # --- Executive Summary values ---
    num_obs = len(final_results.get("L Observed", []))
    num_params = len(final_results.get("X Hat (Final)", []))
    dof = final_results.get("DOF", "")
    apriori = format_number(final_results.get("Apriori Variance", ""))
    aposteriori = format_number(final_results.get("Aposteriori Variance", ""))

    outlier_detected = "Yes" if final_results.get("outliers", False) else "No"
    constraints_used = (
        "Hard and Soft" if (hard_constraints and soft_constraints)
        else "Hard" if hard_constraints
        else "Soft" if soft_constraints
        else "None"
    )

    # --- Observation Equations ---
    eqs = final_results.get("Equations", [])
    L_obs = final_results.get("L Observed", [])
    obs_eqs = []
    for i, (eq, val) in enumerate(zip(eqs, L_obs), start=1):
        obs_eqs.append(f"{i}. {eq} = {format_number(val)}")
    observation_equations = "\n".join(obs_eqs)

    # --- Constraints text ---
    if hard_constraints:
        hard_str = "Hard Constraints:\n" + "\n".join([f"{k}: {v}" for k, v in hard_constraints.items()])
    else:
        hard_str = "Hard Constraints: NOT applicable to this adjustment."
    if soft_constraints:
        soft_str = "Soft Constraints:\n" + "\n".join([f"{k}: {v}" for k, v in soft_constraints.items()])
    else:
        soft_str = "Soft Constraints: NOT applicable to this adjustment."
    constraints_text = f"{hard_str}\n\n{soft_str}"

    # --- Current date/time ---
    curr_date = datetime.today().strftime("%Y-%m-%d")
    curr_time = datetime.today().strftime("%H:%M:%S")

    # --- Build context ---
    tpl = DocxTemplate(template_path)
    context = {
        "num_observations": num_obs,
        "num_params": num_params,
        "dof": dof,
        "apriori_var": apriori,
        "aposteriori_var": aposteriori,
        "outlier_detected": outlier_detected,
        "weight_type": weight_type,
        "constraints_used": constraints_used,
        "observation_equations": observation_equations,
        "constraints": constraints_text,
        "curr_date": curr_date,
        "curr_time": curr_time,
    }

    # --- Graphs (auto-detect input type) ---
    for key, img in [("vtpv_graph", vtpv_graph), ("chi_graph", chi_graph)]:
        img_bytes = _to_image_bytes(img)
        if img_bytes:
            context[key] = InlineImage(tpl, io.BytesIO(img_bytes), width=Inches(4))

    # --- Render & Save DOCX to a buffer ---
    docx_buffer = io.BytesIO()
    tpl.render(context)
    tpl.save(docx_buffer)
    docx_buffer.seek(0)

    # --- Return the DOCX buffer directly ---
    return docx_buffer
