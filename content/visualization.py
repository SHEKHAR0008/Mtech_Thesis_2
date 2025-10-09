# content/visualization.py
import streamlit as st


def visualization_page():
    st.header("📊 Visualization")

    if not st.session_state.get("steps_done", {}).get("adjustment", False):
        st.info("Please run the full analysis from the 'Adjustment' page first.")
        return

    st.subheader("Analysis Graphs")

    # --- FIXED: Use the correct key names from the backend pipeline ---
    plots_info = [
        ("VTPV Convergence", "vtpv_plot"),  # Changed from "vtpv_graph"
        ("Chi-Square Test", "chi_square_plot"),  # Changed from "chi_graph"
        ("Network Plot", "network_plot"),  # This one was already correct
    ]

    for title, key in plots_info:
        st.markdown("---")
        plot_object = st.session_state.get(key)
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            if plot_object:
                with st.popover(f"📊 Show {title}", use_container_width=True):
                    st.plotly_chart(plot_object, use_container_width=True)
            else:
                st.info(f"{title} not available.")
        with col2:
            if plot_object:
                st.download_button(
                    label="⬇️ Download as PNG",
                    data=plot_object.to_image(format="png", scale=3),
                    file_name=f"{key}.png",
                    mime="image/png",
                    key=f"download_{key}",
                    use_container_width=True
                )

    # --- Interactive Error Ellipse Plotting ---
    st.markdown("---")
    st.subheader("Interactive Error Ellipses")

    error_ellipse_plots = st.session_state.get("error_ellipse_plot")
    if "error_ellipse_plot" in st.session_state and st.session_state.error_ellipse_plot:
        error_ellipse_plots = st.session_state.error_ellipse_plot
        error_ellipse_stats = st.session_state.get("error_ellipse_stats", {})

        # Extract station names from plot titles
        station_names = [
            plot.layout.title.text.split(' for ')[-1].split('(')[0].strip()
            for plot in error_ellipse_plots
        ]
        station_map = {name: i for i, name in enumerate(station_names)}

        selected_station = st.selectbox(
            "Select a station to view its error ellipse:",
            station_names,
            key="error_ellipse_select"
        )

        # Single button to show graph + stats
        if st.button("📊 Show Graph & Stats"):
            if selected_station:
                selected_plot = error_ellipse_plots[station_map[selected_station]]

                # Two-column layout: left = plot, right = stats
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.plotly_chart(selected_plot, use_container_width=True)

                with col2:
                    st.markdown("### Station Stats")
                    stats = error_ellipse_stats.get(selected_station, {})
                    if stats:
                        if stats["type"] == "1D":
                            st.write(f"**Coordinate:** {stats['coord']}")
                            st.write(f"**Confidence Interval:** ±{stats['interval']:.4f}")
                        elif stats["type"] == "2D":
                            st.write(f"**Semi-major axis (a):** {stats['a']:.4f}")
                            st.write(f"**Semi-minor axis (b):** {stats['b']:.4f}")
                            st.write(f"**Orientation θ:** {stats['azimuth_deg']:.2f}°")
                        elif stats["type"] == "3D":
                            st.write("**Radii:**")
                            st.write(f"X: {stats['radii'][0]:.4f}")
                            st.write(f"Y: {stats['radii'][1]:.4f}")
                            st.write(f"Z: {stats['radii'][2]:.4f}")
                    else:
                        st.write("Stats not available.")
    else:
        st.info("Error Ellipse Plot not available.")
    st.markdown("---")
    st.session_state["steps_done"]["visualization"] = True
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("⬅️ Previous", use_container_width=True, key="prev_vis"):
            st.session_state.current_step = "adjustment"
            # CRITICAL: When going back, clear all results to allow a fresh run
            keys_to_clear = [
                "final_results", "warning", "dof", "outlier_results", "chi_graph",
                "vtpv_graph", "network_plot", "error_ellipse_plot", "error_ellipse_stats",
                "obs_buffer", "params_buffer", "covar_buffer", "report_buffer",
                "processing_complete"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            # Reset step completion status
            st.session_state["steps_done"]["adjustment"] = False
            st.session_state["steps_done"]["visualization"] = False
            st.session_state["steps_done"]["Download"] = False
            st.rerun()
    with next_col:
        if st.button("Next ➡️", use_container_width=True, key="next_vis"):
            st.session_state.current_step = "Download"
            st.rerun()