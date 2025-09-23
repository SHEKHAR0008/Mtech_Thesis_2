# utils/adjustment_helpers.py

import streamlit as st
import numpy as np


def hard_constraint_ui():
    """
    Renders the UI for adding and managing hard constraints.
    Updates st.session_state.hard_constraints directly.
    """
    st.session_state.soft_constraints = {}

    # Check for existing hard constraints and set a placeholder for the next one
    if "last_hard_constraint" not in st.session_state:
        st.session_state.last_hard_constraint = (0.0, 0.0, 0.0)

    station = st.selectbox(
        "Select a station to fix",
        [s for s in st.session_state.unq_stations if s not in st.session_state.get("hard_constraints", {})],
        key="hc_station_select"
    )

    dim = st.session_state.get("dimension")

    x_val = st.session_state.last_hard_constraint[0]
    y_val = st.session_state.last_hard_constraint[1]
    z_val = st.session_state.last_hard_constraint[2]

    # Use st.text_input as a robust workaround for number input issues
    if dim in ['2D', '3D']:
        x_str = st.text_input("X value", value=str(x_val), key="hc_x")
        y_str = st.text_input("Y value", value=str(y_val), key="hc_y")
    else:
        x_str, y_str = "0.0", "0.0"

    if dim in ['3D', 'Height']:
        z_str = st.text_input("Z value", value=str(z_val), key="hc_z")
    else:
        z_str = "0.0"

    if dim == 'Height':
        st.info("Height adjustment is not yet implemented. This is a placeholder.")

    if st.button("➕ Add Hard Constraint"):
        try:
            # Convert string inputs to floats
            x = float(x_str)
            y = float(y_str)
            z = float(z_str)
        except ValueError:
            st.error("Please enter a valid number for all coordinate fields.")
            return

        if dim == '2D':
            constraint_value = (x, y, 0.0)
        elif dim == '3D':
            constraint_value = (x, y, z)
        elif dim == 'Height':
            constraint_value = (0.0, 0.0, z)
        else:
            st.error("Please select an adjustment type on the main page.")
            return

        # Check for duplicate coordinates
        if constraint_value in st.session_state.hard_constraints.values():
            st.error("This coordinate value is already used by another hard-constrained station.")
            return

        if "hard_constraints" not in st.session_state:
            st.session_state.hard_constraints = {}
        st.session_state.hard_constraints[station] = constraint_value
        st.session_state.last_hard_constraint = constraint_value
        st.success(f"Added hard constraint for {station}: {constraint_value}")
        st.rerun()

    if "hard_constraints" in st.session_state and st.session_state.hard_constraints:
        st.subheader("📌 Current Hard Constraints")
        for sta, xyz in list(st.session_state.hard_constraints.items()):
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(f"**{sta}** → {xyz}")
            with cols[1]:
                if st.button(f"❌", key=f"del_hc_{sta}"):
                    del st.session_state.hard_constraints[sta]
                    st.warning(f"Removed constraint for {sta}")
                    st.rerun()

def soft_constraint_ui(weight_type):
    """
    Renders the UI for adding and managing soft constraints.
    Updates st.session_state.soft_constraints directly.
    """
    st.session_state.hard_constraints = {}

    # Initialize session state for soft constraints and last entered values
    if "soft_constraints" not in st.session_state:
        st.session_state.soft_constraints = {}
    if "last_soft_values" not in st.session_state:
        st.session_state.last_soft_values = (0.0, 0.0, 0.0)

    station = st.selectbox("Select a station for soft constraint", list(st.session_state.unq_stations),
                           key="sc_station_select")

    dim = st.session_state.get("dimension")

    # Check for Height adjustment
    if dim == 'Height':
        st.info("Height adjustment is not yet implemented. This is a placeholder.")
        return

    # Set default values from the last entry
    x_val = st.session_state.last_soft_values[0]
    y_val = st.session_state.last_soft_values[1]
    z_val = st.session_state.last_soft_values[2]

    # Input for coordinates
    st.subheader("Constraint Coordinates")
    if dim in ['2D', '3D']:
        x = st.number_input("X value", value=x_val, format="%.6f", key="sc_x")
        y = st.number_input("Y value", value=y_val, format="%.6f", key="sc_y")
    else:
        x, y = 0.0, 0.0

    if dim == '3D':
        z = st.number_input("Z value", value=z_val, format="%.6f", key="sc_z")
    else:
        z = 0.0

    st.subheader("Enter Variance-Covariance Matrix")

    if dim == '2D':
        matrix_size = 2
    else:  # 3D
        matrix_size = 3

    cov_values = np.zeros((matrix_size, matrix_size))

    if weight_type == "Unity":
        st.info("No input needed. A unity matrix will be used in the adjustment.")
        cov_values = np.eye(matrix_size)

    elif weight_type == "Diagonal":
        st.info("Input for diagonal variances or standard deviations only.")
        mode = st.radio("Input Mode", ["Standard Deviation", "Variance"], key="diag_mode")

        diag_labels = ["X", "Y"] if dim == '2D' else ["X", "Y", "Z"]
        diag_cols = st.columns(matrix_size)
        if mode == "Standard Deviation":
            for i in range(matrix_size):
                sd = diag_cols[i].number_input(f"Std Dev ({diag_labels[i]})", value=1.0, key=f"diag_sd_{i}")
                cov_values[i, i] = sd ** 2
        else:  # Variance
            for i in range(matrix_size):
                var = diag_cols[i].number_input(f"Var({diag_labels[i]})", value=1.0, key=f"diag_var_{i}")
                cov_values[i, i] = var

    elif weight_type == "Full":
        st.info("Enter the full VCV matrix. The upper triangle will be filled via symmetry.")

        for i in range(matrix_size):
            cols = st.columns(matrix_size)
            for j in range(matrix_size):
                key = f"cov_{station}_{i}_{j}"

                if i <= j:
                    if "last_soft_cov_values" in st.session_state and len(
                            st.session_state.last_soft_cov_values) > i and len(
                            st.session_state.last_soft_cov_values[i]) > j:
                        default_val = st.session_state.last_soft_cov_values[i][j]
                    else:
                        default_val = 0.0

                    v = cols[j].number_input(f"Cov[{i + 1},{j + 1}]", value=default_val, key=key)
                    cov_values[i, j] = v
                    if i != j:
                        cov_values[j, i] = v
                else:
                    cols[j].number_input(f"Cov[{i + 1},{j + 1}]", value=cov_values[i, j], disabled=True, key=key)

    if st.button("Add Soft Constraint"):
        # Store coordinates for suggestion
        if dim == '2D':
            constraint_value = (x, y, 0.0)
        else:
            constraint_value = (x, y, z)

        # Check for duplicate coordinates
        # soft_constraints stores values in the format `{"value": (x,y,z), "cov": [...]}`
        for sc_data in st.session_state.soft_constraints.values():
            if sc_data["value"] == constraint_value:
                st.error("This coordinate value is already used by another soft-constrained station.")
                return

        st.session_state.last_soft_values = constraint_value

        # Store VCV for suggestion
        if weight_type != "Unity":
            st.session_state.last_soft_cov_values = cov_values.tolist()
        else:
            st.session_state.last_soft_cov_values = np.eye(matrix_size).tolist()

        # Add to main soft constraints list
        st.session_state.soft_constraints[station] = {"value": constraint_value, "cov": cov_values.tolist()}
        st.success(f"Added soft constraint for {station}")
        st.rerun()

    if "soft_constraints" in st.session_state and st.session_state.soft_constraints:
        st.subheader("📌 Current Soft Constraints")
        for sta, data in list(st.session_state.soft_constraints.items()):
            cols = st.columns([4, 1])
            with cols[0]:
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