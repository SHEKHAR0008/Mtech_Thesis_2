import streamlit as st

def main_page():
    st.markdown("""
    <style>
    /* Overall page styling */
    .main-container {
        text-align: center;
        padding-top: 2rem;
    }

    /* Header styling */
    h1, h2, h3 {
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Button container */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin-top: 3rem;
        flex-wrap: wrap;
    }

    /* Base button style */
    .stButton>button, .custom-disabled-btn {
        width: 15rem !important;
        height: 10rem !important;
        font-size: 1.6rem !important;
        border: 2px solid #4CAF50 !important;
        border-radius: 15px !important;
        transition: all 0.3s ease-in-out !important;
        background: #f8f9fa !important;
        color: #333 !important;
        font-weight: 600 !important;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.08) !important;
        text-align: center;
    }

    /* Hover effect */
    .stButton>button:hover:enabled {
        background-color: #4CAF50 !important;
        color: white !important;
        transform: translateY(-5px);
        box-shadow: 0px 8px 16px rgba(76, 175, 80, 0.4) !important;
    }

    /* Disabled button style */
    .custom-disabled-btn {
        background-color: #e0e0e0 !important;
        color: #888 !important;
        border: 2px dashed #bbb !important;
        cursor: not-allowed !important;
        opacity: 0.7 !important;
        box-shadow: none !important;
    }

    /* Add emoji spacing */
    .button-emoji {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.header("⚙️ Select Adjustment Type")

    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📏\n2D-Baseline", key="2d_button"):
            st.session_state.dimension = '2D'
            st.session_state.current_step = 'data input'
            st.session_state.steps_done = {
                "data input": False,
                "Data validation": False,
                "adjustment": False,
                "visualization": False,
                "Download": False,
            }
            st.rerun()

    with col2:
        if st.button("📐\n3D-Baseline", key="3d_button"):
            st.session_state.dimension = '3D'
            st.session_state.current_step = 'data input'
            st.session_state.steps_done = {
                "data input": False,
                "Data validation": False,
                "adjustment": False,
                "visualization": False,
                "Download": False,
            }
            st.rerun()

    with col3:
        st.markdown(
            """
            <button class="custom-disabled-btn" disabled>
                📊<br>Height<br><small>(Coming soon...)</small>
            </button>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
