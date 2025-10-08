# backend/visualization/network_plot.py (Corrected)

import plotly.graph_objects as go
import numpy as np

def generate_network_plot(final_results, baseline_list):
    """
    Generates an interactive Plotly plot for the 2D network.
    """
    try:
        station_coords = {}
        X_hat_final = final_results.get("X Hat (Final)")
        params_names = final_results.get("PARAMS_Name")
        constants = final_results.get("Constant", {})

        if X_hat_final is not None and params_names is not None:
            for i, param_symbol in enumerate(params_names):
                param_str = str(param_symbol)
                coord_type, station_name = param_str.split("_", 1)
                if station_name not in station_coords:
                    station_coords[station_name] = {'X': None, 'Y': None}
                if coord_type in ['X', 'Y']:
                    station_coords[station_name][coord_type] = X_hat_final[i, 0]

        if constants:
            for const_symbol, value in constants.items():
                const_str = str(const_symbol)
                coord_type, station_name = const_str.split("_", 1)
                if station_name not in station_coords:
                    station_coords[station_name] = {'X': None, 'Y': None}
                if coord_type in ['X', 'Y']:
                    station_coords[station_name][coord_type] = float(value)

        fig = go.Figure()

        # Add baselines as lines
        for baseline in baseline_list:
            from_sta, to_sta = baseline.from_station, baseline.to_station
            if from_sta in station_coords and to_sta in station_coords:
                from_coords = station_coords[from_sta]
                to_coords = station_coords[to_sta]
                fig.add_trace(go.Scatter(
                    x=[from_coords['X'], to_coords['X']],
                    y=[from_coords['Y'], to_coords['Y']],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    hoverinfo='text',
                    text=f"Baseline: {from_sta} to {to_sta}"
                ))

        # Add stations as markers
        station_x = [coords['X'] for coords in station_coords.values()]
        station_y = [coords['Y'] for coords in station_coords.values()]
        station_names = list(station_coords.keys())
        fig.add_trace(go.Scatter(
            x=station_x, y=station_y,
            mode='markers+text',
            text=station_names,
            textposition="top center",
            # --- THIS LINE IS NOW FIXED ---
            marker=dict(symbol='triangle-up', size=10, color='blue'),
            hovertemplate='<b>Station: %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="Interactive Network Plot",
            xaxis_title="X-Coordinate",
            yaxis_title="Y-Coordinate",
            showlegend=False,
            yaxis=dict(scaleanchor="x", scaleratio=1) # Ensures aspect ratio is 1:1
        )
        return fig
    except Exception as e:
        print(f"Error generating interactive Network plot: {e}")
        return None