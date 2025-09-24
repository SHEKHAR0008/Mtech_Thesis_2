from backend.visualization.chi_square_plot import generate_chi_square_plot
from backend.visualization.vtpv_plot import generate_vtpv_plot
from backend.visualization.network_plot import generate_network_plot
from backend.visualization.error_ellipse_plot import plot_interactive_error_ellipses


def generate_plots(final_results, dof, unq_stations, baseline_list, alpha=0.05):
    """
    Generates and returns four plots for geodetic adjustment analysis,
    returning them as buffers for Matplotlib plots and figures for Plotly.
    """
    plots = {}

    # --- 1. Chi-Square Plot (Matplotlib) ---
    plots["chi_square_plot"] = generate_chi_square_plot(final_results, dof, alpha)

    # --- 2. VTPV Convergence Plot (Matplotlib) ---
    plots["vtpv_plot"] = generate_vtpv_plot(final_results)

    # --- 3. Network Plot (Matplotlib) ---
    plots["network_plot"] = generate_network_plot(final_results, baseline_list)

    # --- 4. Error Ellipse Plot (Interactive Plotly Figures) ---
    # This remains a list of Plotly figures as you requested to keep it interactive.
    error_ellipse_figs = plot_interactive_error_ellipses(final_results, conf=0.5)
    plots["error_ellipse_plot"], plots["error_ellipse_stats"] = error_ellipse_figs if len(error_ellipse_figs) > 0 else None

    return plots