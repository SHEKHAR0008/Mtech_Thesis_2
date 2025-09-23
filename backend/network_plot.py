import numpy as np
import matplotlib.pyplot as plt
import io

def generate_network_plot(final_results, baseline_list):
    """
    Generates a static Matplotlib plot for the 2D network plot.
    """
    try:
        def nonlinear_scale(values, method="sqrt"):
            """Apply nonlinear scaling to values."""
            arr = np.array(values, dtype=float)
            sign = np.sign(arr)
            arr = np.abs(arr)
            if method.startswith("sqrt"):
                if ":" in method: n = int(method.split(":")[1])
                else: n = 1
                for _ in range(n): arr = np.sqrt(arr)
                scaled = arr
            elif method == "log":
                scaled = np.log1p(arr)
            elif method == "arcsinh":
                scaled = np.arcsinh(arr)
            elif method == "tanh":
                mean, std = arr.mean(), arr.std() if arr.std() > 0 else 1
                scaled = np.tanh((arr - mean) / std)
            elif method.startswith("power:"):
                try: alpha = float(method.split(":")[1])
                except: alpha = 0.5
                scaled = np.power(arr, alpha)
            else:
                scaled = arr
            return sign * scaled

        station_coords = {}
        X_hat_final = final_results.get("X Hat (Final)")
        params_names = final_results.get("PARAMS_Name")
        constants = final_results.get("Constant", {})

        if X_hat_final is not None and params_names is not None:
            for i, param_symbol in enumerate(params_names):
                param_str = str(param_symbol)
                coord_type, station_name = param_str.split("_", 1)
                if station_name not in station_coords:
                    station_coords[station_name] = {'X': None, 'Y': None, 'Z': None}
                if coord_type in station_coords[station_name]:
                    station_coords[station_name][coord_type] = X_hat_final[i, 0]

        if constants:
            for const_symbol, value in constants.items():
                const_str = str(const_symbol)
                coord_type, station_name = const_str.split("_", 1)
                if station_name not in station_coords:
                    station_coords[station_name] = {'X': None, 'Y': None, 'Z': None}
                if coord_type in station_coords[station_name]:
                    station_coords[station_name][coord_type] = float(value)

        all_x = [coords['X'] for coords in station_coords.values() if coords['X'] is not None]
        all_y = [coords['Y'] for coords in station_coords.values() if coords['Y'] is not None]

        scale_method = "sqrt"
        scaled_station_coords = {}
        if all_x and all_y:
            for station_name, coords in station_coords.items():
                if coords['X'] is not None and coords['Y'] is not None:
                    scaled_station_coords[station_name] = {
                        'X': nonlinear_scale([coords['X']], method=scale_method)[0],
                        'Y': nonlinear_scale([coords['Y']], method=scale_method)[0],
                        'Z': coords['Z']
                    }
                else:
                    scaled_station_coords[station_name] = coords

            xticks_orig = np.linspace(min(all_x), max(all_x), 6)
            yticks_orig = np.linspace(min(all_y), max(all_y), 6)
            xticks_scaled = nonlinear_scale(xticks_orig, method=scale_method)
            yticks_scaled = nonlinear_scale(yticks_orig, method=scale_method)
        else:
            xticks_orig, yticks_orig = [], []
            xticks_scaled, yticks_scaled = [], []
            scaled_station_coords = station_coords

        fig_network, ax_network = plt.subplots(figsize=(10, 8))

        for station_name, coords in scaled_station_coords.items():
            if coords['X'] is not None and coords['Y'] is not None:
                ax_network.scatter(coords['X'], coords['Y'], c='white', s=140,
                                   marker="^", edgecolors="black", zorder=5)
                ax_network.annotate(station_name, (coords['X'], coords['Y']),
                                    textcoords="offset points", xytext=(0, 10), ha='center',
                                    fontsize=10, weight="bold")

        for idx, baseline in enumerate(baseline_list, start=1):
            from_sta, to_sta = baseline.from_station, baseline.to_station
            if from_sta in scaled_station_coords and to_sta in scaled_station_coords:
                from_x, from_y = scaled_station_coords[from_sta]['X'], scaled_station_coords[from_sta]['Y']
                to_x, to_y = scaled_station_coords[to_sta]['X'], scaled_station_coords[to_sta]['Y']
                ax_network.plot([from_x, to_x], [from_y, to_y], 'k-', lw=1, alpha=0.8, zorder=3)
                mid_x, mid_y = (from_x + to_x) / 2, (from_y + to_y) / 2
                ax_network.text(mid_x, mid_y, f"{idx}", fontsize=9, ha='center', va='center',
                                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black"))

        if xticks_scaled.size > 0 and yticks_scaled.size > 0:
            ax_network.set_xticks(xticks_scaled)
            ax_network.set_xticklabels([f"{val:.0f}" for val in xticks_orig])
            ax_network.set_yticks(yticks_scaled)
            ax_network.set_yticklabels([f"{val:.0f}" for val in yticks_orig])

        ax_network.set_title("Network Plot (2D, Nonlinear Scaling)", fontsize=14)
        ax_network.set_xlabel("X-Coordinate (Original Units)", fontsize=12)
        ax_network.set_ylabel("Y-Coordinate (Original Units)", fontsize=12)
        ax_network.grid(True, linestyle="--", alpha=0.6)
        ax_network.set_aspect('equal', adjustable='box')

        buf = io.BytesIO()
        fig_network.savefig(buf, format="png")
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Error generating Network plot: {e}")
        return None