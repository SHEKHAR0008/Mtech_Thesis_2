import streamlit as st


def main_page():
    st.header("Select Adjustment Type")

    st.markdown("""
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
    }
    .stButton>button {
        width: 15rem;
        height: 10rem;
        font-size: 1.5rem;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("2D-Baseline", key="2d_button"):
                st.session_state.dimension = '2D'
                st.session_state.current_step = 'data input'
                # Correctly initialize all steps here
                st.session_state.steps_done = {
                    "data input": False,
                    "Data validation": False,
                    "adjustment": False,
                    "visualization": False,
                    "Download": False,
                }
                st.rerun()
        with col2:
            if st.button("3D-Baseline", key="3d_button"):
                st.session_state.dimension = '3D'
                st.session_state.current_step = 'data input'
                # Correctly initialize all steps here
                st.session_state.steps_done = {
                    "data input": False,
                    "Data validation": False,
                    "adjustment": False,
                    "visualization": False,
                    "Download": False,
                }
                st.rerun()
        with col3:
            st.button("Height", disabled=True, key="height_button", help="This will be implemented soon.")
        st.markdown('</div>', unsafe_allow_html=True)