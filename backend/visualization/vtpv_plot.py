# backend/visualization/vtpv_plot.py

import plotly.graph_objects as go

def generate_vtpv_plot(final_results):
    """
    Generates an interactive Plotly plot for V^T P V convergence.
    """
    try:
        vtpv_values = final_results.get("VTPV_values", [])
        print(vtpv_values)
        if not vtpv_values:
            return None

        iterations = list(range(1, len(vtpv_values) + 1))
        fig = go.Figure(data=go.Scatter(
            x=iterations,
            y=vtpv_values,
            mode='lines+markers',
            hovertemplate='Iteration: %{x}<br>VTPV: %{y:.6f}<extra></extra>'
        ))
        fig.update_layout(
            title="Convergence of V<sup>T</sup>PV",
            xaxis_title="Iteration",
            yaxis_title="V<sup>T</sup>PV",
        )
        return fig
    except Exception as e:
        print(f"Error generating interactive V<sup>T</sup>PV plot: {e}")
        return None