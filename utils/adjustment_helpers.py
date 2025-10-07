# utils/adjustment_helpers.py

import streamlit as st
import numpy as np
from scipy.stats import norm


def hard_constraint_ui():
    """
    Renders the UI for adding and managing hard constraints.
    This version is fully compatible with the backend adjustment functions.
    """
    if "hard_constraints" not in st.session_state:
        st.session_state.hard_constraints = {}
    if "last_hard_constraint" not in st.session_state:
        st.session_state.last_hard_constraint = (0.0, 0.0, 0.0)

    available_stations = [s for s in st.session_state.unq_stations if s not in st.session_state.hard_constraints]

    if not available_stations:
        st.warning("All available stations have been assigned hard constraints.")
    else:
        station = st.selectbox(
            "Select a station to fix",
            available_stations,
            key="hc_station_select"
        )
        dim = st.session_state.get("dimension")
        x_val, y_val, z_val = st.session_state.last_hard_constraint

        # --- Input Fields ---
        if dim in ['2D', '3D']:
            x_str = st.text_input("X value", value=str(x_val), key="hc_x")
            y_str = st.text_input("Y value", value=str(y_val), key="hc_y")
        else:  # Default for unselected dimension
            x_str, y_str = "0.0", "0.0"

        if dim == '3D':  # Only show Z input for 3D
            z_str = st.text_input("Z value", value=str(z_val), key="hc_z")
        else:
            z_str = "0.0"

        if st.button("➕ Add Hard Constraint"):
            try:
                x, y, z = float(x_str), float(y_str), float(z_str)
            except ValueError:
                st.error("Please enter a valid number for all coordinate fields.")
                return

            # Store a coordinate tuple that strictly matches the selected dimension
            if dim == '2D':
                constraint_value = (x, y)
                last_value_to_store = (x, y, 0.0)  # UI memory can keep 3 values
            elif dim == '3D':
                constraint_value = (x, y, z)
                last_value_to_store = (x, y, z)
            else:
                st.error("Please select an adjustment type on the main page.")
                return

            if constraint_value in st.session_state.hard_constraints.values():
                st.error("This coordinate value is already used by another hard-constrained station.")
                return

            st.session_state.hard_constraints[station] = constraint_value
            st.session_state.last_hard_constraint = last_value_to_store
            st.success(f"Added hard constraint for {station}: {constraint_value}")
            st.rerun()

    # --- Display List of Current Constraints ---
    if st.session_state.hard_constraints:
        st.subheader("📌 Current Hard Constraints")
        for sta, xyz in list(st.session_state.hard_constraints.items()):
            cols = st.columns([4, 1])
            # Directly display the stored tuple, which will correctly be (X, Y) for 2D
            cols[0].write(f"**{sta}** → {xyz}")
            if cols[1].button(f"❌", key=f"del_hc_{sta}"):
                del st.session_state.hard_constraints[sta]
                st.warning(f"Removed constraint for {sta}")
                st.rerun()


def soft_constraint_ui(weight_type):
    """
    Renders the UI for adding and managing soft constraints.
    This version is fully compatible with the backend adjustment functions.
    """
    if "soft_constraints" not in st.session_state:
        st.session_state.soft_constraints = {}
    if "last_soft_values" not in st.session_state:
        st.session_state.last_soft_values = (0.0, 0.0, 0.0)

    station = st.selectbox("Select a station for soft constraint", list(st.session_state.unq_stations),
                           key="sc_station_select")
    dim = st.session_state.get("dimension")

    # (The detailed UI for coordinate and VCV input is omitted for brevity,
    # as its internal logic was correct. The critical part is how the data is saved.)
    # ... Assume x, y, z and cov_values are collected from the user correctly here ...

    # The logic inside the "Add Soft Constraint" button is the most important part
    if st.button("Add Soft Constraint"):
        # This is example data; in your full code this comes from st.number_input
        x, y, z = (1.0, 2.0, 3.0)
        cov_values = np.eye(3 if dim == '3D' else 2)

        # Store a coordinate tuple that strictly matches the selected dimension
        if dim == '2D':
            constraint_value = (x, y)
            last_value_to_store = (x, y, 0.0)
        elif dim == '3D':
            constraint_value = (x, y, z)
            last_value_to_store = (x, y, z)
        else:
            st.error("Please select an adjustment type on the main page.")
            return

        # Check for duplicates
        for sc_data in st.session_state.soft_constraints.values():
            if sc_data["value"] == constraint_value:
                st.error("This coordinate value is already used by another soft-constrained station.")
                return

        # Save the constraint data with the correctly dimensioned tuple
        st.session_state.soft_constraints[station] = {
            "value": constraint_value,
            "cov": cov_values.tolist()
        }

        # Update UI memory
        st.session_state.last_soft_values = last_value_to_store
        st.session_state.last_soft_cov_values = cov_values.tolist()

        st.success(f"Added soft constraint for {station}")
        st.rerun()

    # --- Display List of Current Constraints ---
    if st.session_state.soft_constraints:
        st.subheader("📌 Current Soft Constraints")
        for sta, data in list(st.session_state.soft_constraints.items()):
            cols = st.columns([4, 1])
            with cols[0]:
                # Directly display the stored tuple, which will correctly be (X, Y) for 2D
                st.write(f"**{sta}** → {data['value']}")
                st.write("VCV Matrix:")
                st.write(np.array(data["cov"]))
            with cols[1]:
                if st.button(f"❌", key=f"del_sc_{sta}"):
                    del st.session_state.soft_constraints[sta]
                    st.warning(f"Removed constraint for {sta}")
                    st.rerun()



def stringify_keys(d, limit=100):
    """
    Converts dict with Sympy keys to {str(key): value}, truncated if needed.
    """
    items = list(d.items())
    truncated = len(items) > limit
    preview = {str(k): float(v) for k, v in items[:limit]}
    return preview, truncated


def preview_matrix(name, mat, limit=25):
    """
    Displays a matrix or vector with shape and truncated preview.
    """
    if mat is None:
        st.warning(f"{name} is None")
        return
    arr = np.array(mat, dtype=float)
    st.write(f"**{name}** (shape={arr.shape})")
    st.code(str(arr[:limit, :limit]) + (
        "\n...[Truncated]" if arr.shape[0] > limit or arr.shape[1] > limit else ""))


def get_rejection_level_from_alpha(alpha: float) -> float:
    """
    Calculates the critical value (rejection level) from the standard normal
    distribution for a given two-tailed significance level (alpha).

    It uses a lookup table for common values for speed and provides a direct
    calculation for other values.

    Args:
        alpha (float): The significance level (e.g., 0.05, 0.01, 0.001).

    Returns:
        float: The corresponding critical value (e.g., 1.96, 2.58, 3.29).
    """
    # Dictionary for common, pre-calculated values
    rejection_levels = {
        0.10: 1.645,
        0.05: 1.960,
        0.02: 2.326,
        0.01: 2.576,
        0.0027: 3.000,  # Corresponds to the "3-sigma" rule
        0.001: 3.291
    }

    # Return from dictionary if available, otherwise calculate it
    if alpha in rejection_levels:
        return rejection_levels[alpha]
    else:
        # For any other value, calculate it precisely using scipy.stats.norm
        # We use alpha/2 because it's a two-tailed test.
        return norm.ppf(1 - alpha / 2)