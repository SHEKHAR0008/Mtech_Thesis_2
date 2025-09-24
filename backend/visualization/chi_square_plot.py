import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import io

def generate_chi_square_plot(final_results, dof, alpha=0.05):
    """
    Generates a static Matplotlib plot for the Chi-Square variance test.
    """
    try:
        sigma_not_square = final_results.get("Apriori Variance", None)
        sigma_not_hat_squared = final_results.get("Aposteriori Variance", None)

        if sigma_not_square is None or sigma_not_hat_squared is None:
            raise ValueError("final_results must contain 'Apriori Variance' and 'Aposteriori Variance'.")

        r = dof
        chi2_statistic = float((r * sigma_not_hat_squared) / sigma_not_square)

        chi2_lower = chi2.ppf(alpha / 2, r)
        chi2_upper = chi2.ppf(1 - alpha / 2, r)

        if chi2_lower <= chi2_statistic <= chi2_upper:
            decision = "Accept null hypothesis (no significant difference)."
        else:
            decision = "Reject null hypothesis (significant difference)."

        x = np.linspace(0, chi2.ppf(0.999, r), 1000)
        y = chi2.pdf(x, r)

        fig_chi, ax_chi = plt.subplots(figsize=(10, 6))
        ax_chi.plot(x, y, label=f"Chi-Square Distribution (dof={r})", color="blue")
        ax_chi.fill_between(x, 0, y, where=(x <= chi2_lower), color="red", alpha=0.5, label="Rejection Region (Lower)")
        ax_chi.fill_between(x, 0, y, where=(x >= chi2_upper), color="red", alpha=0.5, label="Rejection Region (Upper)")
        ax_chi.fill_between(x, 0, y, where=(chi2_lower < x) & (x < chi2_upper), color="green", alpha=0.5,
                            label="Acceptance Region")
        ax_chi.axvline(chi2_lower, color="black", linestyle="--", label=f"Lower Critical Value ({chi2_lower:.2f})")
        ax_chi.axvline(chi2_upper, color="black", linestyle="--", label=f"Upper Critical Value ({chi2_upper:.2f})")
        ax_chi.axvline(chi2_statistic, color="purple", linestyle="-",
                       label=f"Chi-Square Statistic ({chi2_statistic:.2f})")
        ax_chi.set_title("Chi-Square Test: Rejection and Acceptance Regions", fontsize=14)
        ax_chi.set_xlabel("Chi-Square Value", fontsize=12)
        ax_chi.set_ylabel("Probability Density", fontsize=12)
        ax_chi.legend(fontsize=10)
        ax_chi.grid(alpha=0.3)

        buf = io.BytesIO()
        fig_chi.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Error generating Chi-Square plot: {e}")
        return None