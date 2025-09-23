import numpy as np
from collections import defaultdict, deque
import streamlit as st

def initial_guess(
    baselines,
    params,
    hard_constraints=None,
    soft_constraints=None,
    reference=None,
):
    """
    Build initial guess values and X_hat.
    """
    hard_constraints = hard_constraints or {}
    soft_constraints = soft_constraints or {}
    dim = 2 if st.session_state.dimension == '2D' else 3

    coords = defaultdict(list)

    # --- Seeding ---
    if hard_constraints or soft_constraints:
        for sta, xyz in hard_constraints.items():
            coords[sta].append(tuple(map(float, xyz)))
        for sta, data in soft_constraints.items():
            coords[sta].append(tuple(map(float, data["value"])))
    else:
        if reference is None:
            reference = next(iter(params.keys()))
        if dim == 2:
            coords[reference].append((0.0, 0.0))
        else:
            coords[reference].append((0.0, 0.0, 0.0))

    # --- Build adjacency graph ---
    graph = {}
    for b in baselines:
        vec = np.asarray(b.vector, dtype=float).reshape(dim,)
        graph.setdefault(b.from_station, []).append((b.to_station, vec))
        graph.setdefault(b.to_station, []).append((b.from_station, -vec))

    # --- BFS propagation ---
    queue = deque(coords.keys())
    visited = set()
    while queue:
        sta = queue.popleft()
        for neigh, vec in graph.get(sta, []):
            for base_xyz in coords[sta]:
                guess = tuple(np.array(base_xyz) + vec)
                coords[neigh].append(guess)
            if neigh not in visited:
                queue.append(neigh)
        visited.add(sta)

    # --- Average multiple path estimates ---
    final_coords = {}
    for sta, guesses in coords.items():
        arr = np.array(guesses, dtype=float)
        if arr.size:
            final_coords[sta] = tuple(arr.mean(axis=0))

    # --- Soft constraints override propagation ---
    for sta, data in soft_constraints.items():
        final_coords[sta] = tuple(map(float, data["value"]))

    # --- Build values + new_params + X_hat ---
    values = {}
    new_params = {}
    X_list = []
    for sta, (X_sym, Y_sym, Z_sym) in params.items():
        if sta in hard_constraints:
            continue

        new_params[sta] = (X_sym, Y_sym) if dim == 2 else (X_sym, Y_sym, Z_sym)
        if sta in final_coords:
            if dim == 2:
                x0, y0 = final_coords[sta]
                values[X_sym] = float(x0)
                values[Y_sym] = float(y0)
                X_list.extend([float(x0), float(y0)])
            else:
                x0, y0, z0 = final_coords[sta]
                values[X_sym] = float(x0)
                values[Y_sym] = float(y0)
                values[Z_sym] = float(z0)
                X_list.extend([float(x0), float(y0), float(z0)])
        else:
            if dim == 2:
                values[X_sym] = 0.0
                values[Y_sym] = 0.0
                X_list.extend([0.0, 0.0])
            else:
                values[X_sym] = 0.0
                values[Y_sym] = 0.0
                values[Z_sym] = 0.0
                X_list.extend([0.0, 0.0, 0.0])

    X_hat = np.array(X_list, dtype=float).reshape(-1, 1)

    # --- Build constants dict for hard constraints ---
    constants = {}
    for sta, xyz in hard_constraints.items():
        X_sym, Y_sym, Z_sym = params[sta]
        constants[X_sym] = float(xyz[0])
        constants[Y_sym] = float(xyz[1])
        if dim == 3:
            constants[Z_sym] = float(xyz[2])

    return values, X_hat, constants, new_params