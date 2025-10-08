# backend/visualization/chi_square_plot.py

import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2


def generate_chi_square_plot(final_results, dof, alpha=0.05):
    """
    Generates an interactive Plotly plot for the Chi-Square variance test.
    """
    try:
        sigma_not_square = final_results.get("Apriori Variance", 1.0)
        sigma_not_hat_squared = final_results.get("Aposteriori Variance", 1.0)
        if isinstance(sigma_not_hat_squared, (list, np.ndarray)):
            while isinstance(sigma_not_hat_squared, (list, np.ndarray)):
                sigma_not_hat_squared = sigma_not_hat_squared[0]
        r = dof
        print(sigma_not_square,sigma_not_hat_squared,r)

        # Ensure values are valid for calculation
        if not all(isinstance(val, (int, float)) and val > 0 for val in [sigma_not_square, sigma_not_hat_squared, r]):
            print("HI")
            return None

        chi2_statistic = float((r * sigma_not_hat_squared) / sigma_not_square)
        chi2_lower = chi2.ppf(alpha / 2, r)
        chi2_upper = chi2.ppf(1 - alpha / 2, r)

        x = np.linspace(chi2.ppf(0.001, r), chi2.ppf(0.999, r), 500)
        y = chi2.pdf(x, r)

        fig = go.Figure()

        # Main distribution curve
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'Chi-Square PDF (dof={r})', line_color='royalblue'))

        # Acceptance region (green)
        fig.add_trace(go.Scatter(
            x=np.concatenate([[chi2_lower], x[(x > chi2_lower) & (x < chi2_upper)], [chi2_upper]]),
            y=np.concatenate([[0], y[(x > chi2_lower) & (x < chi2_upper)], [0]]),
            fill='tozeroy', mode='none', fillcolor='rgba(0, 171, 56, 0.5)', name='Acceptance Region'
        ))

        # Rejection regions (red)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x[x < chi2_lower], [chi2_lower]]),
            y=np.concatenate([y[x < chi2_lower], [0]]),
            fill='tozeroy', mode='none', fillcolor='rgba(234, 67, 53, 0.5)', name='Rejection Region'
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([[chi2_upper], x[x > chi2_upper]]),
            y=np.concatenate([[0], y[x > chi2_upper]]),
            fill='tozeroy', mode='none', fillcolor='rgba(234, 67, 53, 0.5)', showlegend=False
        ))

        # Vertical lines for critical values and statistic
        fig.add_vline(x=chi2_lower, line_dash="dash", line_color="black", annotation_text=f"Lower: {chi2_lower:.2f}")
        fig.add_vline(x=chi2_upper, line_dash="dash", line_color="black", annotation_text=f"Upper: {chi2_upper:.2f}")
        fig.add_vline(x=chi2_statistic, line_width=3, line_color="purple",
                      annotation_text=f"Statistic: {chi2_statistic:.2f}")

        fig.update_layout(
            title="Interactive Chi-Square Test",
            xaxis_title="Chi-Square Value",
            yaxis_title="Probability Density",
            legend_title_text='Regions'
        )
        return fig
    except Exception as e:
        print(f"Error generating interactive Chi-Square plot: {e}")
        return None