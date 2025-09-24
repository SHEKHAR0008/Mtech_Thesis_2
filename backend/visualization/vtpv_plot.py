import numpy as np
import matplotlib.pyplot as plt
import io

def generate_vtpv_plot(final_results):
    """
    Generates a static Matplotlib plot for V^T P V convergence.
    """
    try:
        vtpv_values = final_results.get("VTPV_values", [])
        if not vtpv_values:
            return None
        else:
            fig_vtpv, ax_vtpv = plt.subplots(figsize=(8, 5))
            ax_vtpv.plot(range(1, len(vtpv_values) + 1), vtpv_values, marker="o", linestyle="-", color="b")
            ax_vtpv.set_xlabel("Iteration")
            ax_vtpv.set_ylabel("V^T P V")
            ax_vtpv.set_title("Convergence of V^T P V")
            ax_vtpv.grid(True)

            buf = io.BytesIO()
            fig_vtpv.savefig(buf, format="png")
            buf.seek(0)
            return buf
    except Exception as e:
        print(f"Error generating VTPV plot: {e}")
        return None