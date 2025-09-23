import sympy as sp
import numpy as np
import streamlit as st
from typing import List

class Baseline:
    def __init__(self, baseline_id, from_sta, to_sta, vector, vcv):
        self.baseline_id = baseline_id
        self.from_station = from_sta
        self.to_station = to_sta
        self.vector = np.array(vector, dtype=float)
        self.vcv = np.array(vcv, dtype=float)

    def __repr__(self):
        return f"{self.baseline_id}: {self.from_station}->{self.to_station} | {self.vector}\n"


def build_observation_system(baselines: List[Baseline], weight_type):
    """
    Build observation equations, symbols, and weight matrix from parsed baselines.
    """
    stations = sorted({b.from_station for b in baselines} | {b.to_station for b in baselines})

    # Define parameters based on dimension
    if st.session_state.dimension == '2D':
        params = {s: (sp.Symbol(f"X_{s}"), sp.Symbol(f"Y_{s}")) for s in stations}
    else:  # 3D
        params = {s: (sp.Symbol(f"X_{s}"), sp.Symbol(f"Y_{s}"), sp.Symbol(f"Z_{s}")) for s in stations}

    obs_vec = []
    equations = []
    labels = []

    for b in baselines:
        # Conditional unpacking based on dimension
        if st.session_state.dimension == '2D':
            Xh, Yh = params[b.from_station]
            Xt, Yt = params[b.to_station]
            dx, dy = b.vector[0], b.vector[1]

            eqx = Xt - Xh
            eqy = Yt - Yh

            equations.extend([eqx, eqy])
            obs_vec.extend([dx, dy])
            labels.extend([f"DelX_{b.from_station}_{b.to_station}", f"DelY_{b.from_station}_{b.to_station}"])

        else:  # 3D
            Xh, Yh, Zh = params[b.from_station]
            Xt, Yt, Zt = params[b.to_station]
            dx, dy, dz = b.vector[0], b.vector[1], b.vector[2]

            eqx = Xt - Xh
            eqy = Yt - Yh
            eqz = Zt - Zh

            equations.extend([eqx, eqy, eqz])
            obs_vec.extend([dx, dy, dz])
            labels.extend([f"DelX_{b.from_station}_{b.to_station}", f"DelY_{b.from_station}_{b.to_station}",
                           f"DelZ_{b.from_station}_{b.to_station}"])

    obs_vec = np.array(obs_vec, dtype=float).reshape(-1, 1)

    # --- Build Weight Matrix ---
    n = len(obs_vec)
    dim = 2 if st.session_state.dimension == '2D' else 3

    if weight_type == "unity":
        P = np.eye(n)
    elif weight_type == "diagonal":
        diag_entries = []
        for b in baselines:
            diag_entries.extend([1 / b.vcv[i, i] for i in range(dim)])
        P = np.diag(diag_entries)
    elif weight_type == "full":
        blocks = [np.linalg.inv(b.vcv) for b in baselines]
        P = np.zeros((n, n))
        for i, block in enumerate(blocks):
            P[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = block
    else:
        raise ValueError("weight_type must be 'unity', 'diagonal', or 'full'")

    # --- Meta information ---
    N = n
    U = len(stations) * dim
    dof = N - U
    # print(P)

    return obs_vec, equations, params, labels, P, N, U, dof