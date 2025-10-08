# content/visualization.py

import streamlit as st
import io
import json
from backend.plots import generate_plots

@st.dialog("Plot Viewer")
def view_plot(title, plot_object):
    """Displays a Plotly chart in a Streamlit dialog."""
    st.header(title)
    st.plotly_chart(plot_object, use_container_width=True)

def visualization_page():
    """
    Renders the Visualization page, displaying all generated plots.
    """
    st.header("Visualization")

    if not st.session_state.get("steps_done", {}).get("adjustment", False):
        st.info("Please complete Adjustment first.")
        return

    # Check if plots are already generated and cached
    if not st.session_state.get("chi_graph"):
        try:
            plots= generate_plots(
                st.session_state.final_results,
                st.session_state.dof,
                st.session_state.unq_stations,
                st.session_state.baseline_list,
            )
            st.session_state.chi_graph = plots["chi_square_plot"]
            st.session_state.vtpv_graph = plots["vtpv_plot"]
            st.session_state.network_plot = plots["network_plot"]
            st.session_state.error_ellipse_plot = plots["error_ellipse_plot"]
            st.session_state.error_ellipse_stats = plots["error_ellipse_stats"]
        except Exception as e:
            st.error(f"Failed to generate plots: {e}")
            st.session_state.chi_graph = None
            st.session_state.vtpv_graph = None
            st.session_state.network_plot = None
            st.session_state.error_ellipse_plot = None

    st.subheader("Analysis Graphs")

    # Define the interactive plots to be displayed
    plots_info = [
        ("VTPV Convergence", "vtpv_graph"),
        ("Chi-Square Test", "chi_graph"),
        ("Network Plot", "network_plot"),
    ]

    # for title, key in plots_info:
    #     st.markdown("---")
    #     # Get the Plotly figure object from session state
    #     plot_object = st.session_state.get(key)
    #
    #     # Create two columns for the "Show" and "Download" buttons
    #     col1, col2 = st.columns([0.7, 0.3])
    #
    #     with col1:
    #         if plot_object:
    #             # Use st.popover to create a button that reveals the plot
    #             with st.popover(f"📊 Show {title}", use_container_width=True):
    #                 # Display the interactive Plotly chart inside the popover
    #                 st.plotly_chart(plot_object, use_container_width=True)
    #         else:
    #             st.info(f"{title} not available.")
    #
    #     with col2:
    #         if plot_object:
    #             # Create a download button for the plot as a static PNG image
    #             st.download_button(
    #                 label="⬇️ Download as PNG",
    #                 # Convert the Plotly figure to PNG bytes on the fly
    #                 data=plot_object.to_image(format="png", scale=3),  # scale=3 for high resolution
    #                 file_name=f"{key}.png",
    #                 mime="image/png",
    #                 key=f"download_{key}",
    #                 use_container_width=True
    #             )
    #         else:
    #             # Keep the layout consistent by using an empty placeholder
    #             st.empty()
    for title, key in plots_info:
        st.markdown("---")
        # Get the plot object from the unified 'plots' dictionary
        plot_object = st.session_state.get(key)

        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            if plot_object:
                # --- STEP 2: Call the dialog function when the button is clicked ---
                if st.button(f"🔎 View {title}", use_container_width=True, key=f"view_{key}"):
                    view_plot(title, plot_object)  # This now opens the dialog
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
            else:
                st.empty()
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
                            st.write(f"**Orientation θ:** {stats['theta_deg']:.2f}°")
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
        if st.button("⬅️ Previous", width='stretch', key="prev_vis"):
            st.session_state.current_step = "adjustment"
            st.session_state.viewed_results = False
            st.session_state.chi_graph = None
            st.session_state.vtpv_graph = None
            st.session_state.network_plot = None
            st.session_state.error_ellipse_plot = None
            st.session_state["steps_done"]["visualization"] = False
            st.rerun()
    with next_col:
        if st.button("Next ➡️", width='stretch', key="next_vis"):
            st.session_state.current_step = "Download"
            st.session_state.viewed_results = False
            st.rerun()