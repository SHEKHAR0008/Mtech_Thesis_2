import numpy as np
import sympy as sp
import streamlit as st


def apply_constraints(obs_vec, equations, params, P,
                      labels,
                      hard_constraints=None,
                      soft_constraints=None,
                      weight_type="unity"):
    """
    Applies hard and soft constraints to a system of observation equations.

    - Hard constraints eliminate parameters by substituting their known values.
    - Soft constraints add new pseudo-observations to the system.

    Returns the updated system components and metadata (dof, etc.).
    """
    hard_constraints = hard_constraints or {}
    soft_constraints = soft_constraints or {}
    dim = 2 if st.session_state.dimension == '2D' else 3

    new_obs_vec = obs_vec.copy()
    new_eqs = list(equations)

    # Deconstruct P into blocks for easier reassembly later
    num_baselines = len(obs_vec) // dim
    P_blocks = [P[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] for i in range(num_baselines)]

    # ----- Apply Hard Constraints -----
    # Hard constraints reduce the number of unknowns. We substitute their known
    # values into the equations and move the resulting constants to the observation vector.
    if hard_constraints:
        updated_obs = []
        updated_eqs = []

        for eq, obs in zip(new_eqs, new_obs_vec.flatten()):
            shift_val = 0.0
            reduced_eq = eq

            for sta, xyz in hard_constraints.items():
                # FIX: Unpack symbols correctly based on dimension to avoid crashing in 2D
                syms = params[sta]

                # Create a mapping from symbol to its fixed numerical value
                subs_map = {syms[i]: xyz[i] for i in range(dim)}

                for sym, val in subs_map.items():
                    if sym in reduced_eq.free_symbols:
                        coeff = reduced_eq.coeff(sym)
                        # This moves the constant term (coeff * known_value) to the right side
                        shift_val += val * float(coeff)
                        reduced_eq = reduced_eq.subs(sym, 0)

            updated_eqs.append(sp.simplify(reduced_eq))
            updated_obs.append(obs - shift_val)

        new_eqs = updated_eqs
        new_obs_vec = np.array(updated_obs, dtype=float).reshape(-1, 1)

    # ----- Apply Soft Constraints -----
    # Soft constraints add new pseudo-observations to the system.
    if soft_constraints:
        for sta, data in soft_constraints.items():
            if sta not in params:
                raise ValueError(f"Soft constraint station '{sta}' not found in parameters.")

            # REFINEMENT: Use slicing and the 'dim' variable for cleaner code
            syms_to_add = params[sta][:dim]
            obs_to_add = data["value"][:dim]
            cov_to_use = np.array(data["cov"], dtype=float)[:dim, :dim]

            new_eqs.extend(syms_to_add)
            new_obs_vec = np.vstack([new_obs_vec, np.array(obs_to_add).reshape(-1, 1)])
            labels.extend([f"{str(s)}_soft" for s in syms_to_add])

            # Select the appropriate weight block based on the sliced covariance
            if weight_type == "unity":
                block = np.eye(dim)
            elif weight_type == "diagonal":
                block = np.diag(1 / np.diag(cov_to_use))
            elif weight_type == "full":
                block = np.linalg.inv(cov_to_use)
            else:
                block = np.eye(dim)  # Default to unity
            P_blocks.append(block)

    # Reassemble the final weight matrix P from all blocks
    n_total_obs_sets = len(P_blocks)
    new_P = np.zeros((n_total_obs_sets * dim, n_total_obs_sets * dim))
    for i, blk in enumerate(P_blocks):
        new_P[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = blk

    # Update metadata for the final system
    N = len(new_obs_vec)

    # FIX: Calculate the number of unknowns (U) correctly by subtracting fixed stations
    num_unknown_stations = len(params) - len(hard_constraints)
    U = num_unknown_stations * dim

    dof = N - U

    return new_obs_vec, new_eqs, new_P, labels, N, U, dof