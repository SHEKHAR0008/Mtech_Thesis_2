# utils/ui_helpers.py

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

def _go_to_step(step_name):
    """
    Callback function to update the current step and viewed_results flag.
    """
    st.session_state.current_step = step_name
    st.session_state.viewed_results = False

def setup_navigation_buttons(pages, current_step, steps_done):
    """
    Renders the sidebar navigation buttons with a reliable color-coding scheme
    using the streamlit-extras library for consistent styling.

    Args:
        pages (list): A list of step names in sequential order.
        current_step (str): The name of the currently active step.
        steps_done (dict): A dictionary tracking the completion status of each step.
    """
    st.markdown("""
            <style>
            .arrow-divider {
                display: flex;
                justify-content: center;
                font-size: 18px;
                color: #666;
                margin: -2px 0;
            }
            /* Your other button CSS would go here */
            </style>
        """, unsafe_allow_html=True)
    st.markdown("## 📌 Navigation")

    current_step_index = pages.index(current_step)

    for i, p in enumerate(pages):
        is_active = (current_step == p)
        is_done = steps_done.get(p, False)

        # Determine the button's CSS style based on its state
        button_style = ""
        if is_active:
            button_style = """
                button {
                    font-weight: 600;
                    background-color: #2e7dff;
                    color: white;
                    border: 1px solid #2e7dff;
                    border-radius: 6px;
                    min-height: 44px;
                }
            button:hover {
                transform: scale(1.02);
            }
            """
        elif is_done:
            button_style = """
                button {
                    font-weight: 600;
                    background-color: #28a745;
                    color: white;
                    border: 1px solid #28a745;
                    border-radius: 6px;
                    min-height: 44px;
                }
            button:hover {
                transform: scale(1.02);
            }
            """
        else:
            button_style = """
                button {
                    font-weight: 600;
                    background-color: #f0f2f6;
                    color: #555;
                    border: 1px solid #d9d9d9;
                    border-radius: 6px;
                    min-height: 44px;
                }
            button:hover {
                transform: scale(1.02);
            }
            """

        # A button is enabled if it's a completed step or the very next one
        is_enabled = is_done or is_active or (i == current_step_index + 1)
        if i == 0:
            is_enabled = True # Always enable the first step

        # Use a stylable_container to wrap each button and apply the specific CSS
        with stylable_container(
            key=f"container_{p}",
            css_styles=button_style
        ):
            st.button(
                p,
                key=f"nav_{p}",
                use_container_width=True,
                disabled=not is_enabled,
                on_click=_go_to_step,
                args=[p]
            )

        # Add arrow after each except last
        if i < len(pages) - 1:
            st.markdown('<div class="arrow-divider">⬇</div>', unsafe_allow_html=True)