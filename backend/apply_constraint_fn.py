import numpy as np
import sympy as sp
import streamlit as st

def apply_constraints(obs_vec, equations, params, P,
                      labels,
                      hard_constraints=None,
                      soft_constraints=None,
                      weight_type="unity"):
    """
    Apply soft constraints (pseudo-observations) on top of baseline system.
    Hard constraints are ignored here (handled later in initial_guess).

    Returns updated obs_vec, equations, P, N, U, dof
    """
    dim = 2 if st.session_state.dimension == '2D' else 3

    new_obs_vec = obs_vec.copy()
    new_eqs = list(equations)
    new_params = dict(params)
    P_blocks = [P[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] for i in range(len(obs_vec) // dim)]

    # ----- Hard constraints -----
    if hard_constraints:
        updated_obs = []
        updated_eqs = []

        for eq, obs in zip(new_eqs, new_obs_vec.flatten()):
            shift_val = 0.0
            reduced_eq = eq

            for sta, xyz in hard_constraints.items():
                Xs, Ys, Zs = params[sta]
                subs_map = {Xs: xyz[0], Ys: xyz[1]}
                if dim == 3:
                    subs_map[Zs] = xyz[2]

                for sym, val in subs_map.items():
                    if sym in reduced_eq.free_symbols:
                        coeff = reduced_eq.coeff(sym)
                        if coeff > 0:
                            shift_val += val * float(coeff)
                        elif coeff < 0:
                            shift_val -= val * float(abs(coeff))
                        reduced_eq = reduced_eq.subs(sym, 0)

            updated_eqs.append(sp.simplify(reduced_eq))
            updated_obs.append(obs - shift_val)

        new_eqs = updated_eqs
        new_obs_vec = np.array(updated_obs, dtype=float).reshape(-1, 1)

    # ----- Soft constraints only -----
    if soft_constraints:
        for sta, data in soft_constraints.items():
            if sta not in params:
                raise ValueError(f"Soft constraint station '{sta}' not found in parameters.")

            Xs, Ys = params[sta][0], params[sta][1]
            x0, y0 = data["value"][0], data["value"][1]
            cov = np.array(data["cov"], dtype=float)

            if dim == 3:
                Zs = params[sta][2]
                z0 = data["value"][2]
                new_eqs.extend([Xs, Ys, Zs])
                new_obs_vec = np.vstack([new_obs_vec, [[x0], [y0], [z0]]])
                labels.extend([f"X_{sta}", f"Y_{sta}", f"Z_{sta}"])
            else:
                new_eqs.extend([Xs, Ys])
                new_obs_vec = np.vstack([new_obs_vec, [[x0], [y0]]])
                labels.extend([f"X_{sta}", f"Y_{sta}"])

            if weight_type == "unity":
                block = np.eye(dim)
            elif weight_type == "diagonal":
                if dim == 2:
                    block = np.diag([1 / cov[0, 0], 1 / cov[1, 1]])
                else:
                    block = np.diag([1 / cov[0, 0], 1 / cov[1, 1], 1 / cov[2, 2]])
            elif weight_type == "full":
                block = np.linalg.inv(cov)
            else:
                block = np.eye(dim)
            P_blocks.append(block)

    # Reassemble P
    n_blocks = len(P_blocks)
    new_P = np.zeros((n_blocks * dim, n_blocks * dim))
    for i, blk in enumerate(P_blocks):
        new_P[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = blk

    # Update meta
    N = len(new_obs_vec)
    U = len(new_params) * dim
    dof = N - U

    return new_obs_vec, new_eqs, new_P,labels, N, U, dof