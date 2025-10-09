import numpy as np
from collections import defaultdict, deque
import sympy
import streamlit as st


# # Mocking Streamlit's session_state for a runnable standalone example
# class MockSessionState:
#     def __init__(self, dimension='2D'):
#         self.dimension = dimension
#
#
# st = type('Streamlit', (), {'session_state': MockSessionState()})()


def initial_guess(
        baselines,
        params: dict,
        hard_constraints: dict = None,
        soft_constraints: dict = None,
        reference: str = None,
):
    """
    Builds initial coordinate guesses (X_hat) for a least squares adjustment.

    This function propagates coordinates through a network from known points (constraints)
    or a reference point, averages estimates from multiple paths, and formats the
    output for the adjustment engine. It is dimension-aware and handles 2D/3D cases.

    Returns:
        A tuple of (values, X_hat, constants, new_params).
    """
    hard_constraints = hard_constraints or {}
    soft_constraints = soft_constraints or {}

    # 1. Determine dimension and validate input
    if st.session_state.dimension == '2D':
        dim = 2
    elif st.session_state.dimension == '3D':
        dim = 3
    else:
        raise ValueError("Dimension must be '2D' or '3D'")

    for b in baselines:
        if len(b.vector) != dim:
            raise ValueError(
                f"Dimension mismatch: Session state is '{st.session_state.dimension}' but baseline "
                f"'{b.baseline_id}' has a {len(b.vector)}D vector."
            )

    # 2. Seed initial coordinates from constraints or a reference point
    coords = defaultdict(list)
    seed_points = set()

    # Use hard and soft constraints as starting points
    for sta, xyz in hard_constraints.items():
        coords[sta].append(tuple(map(float, xyz)))
        seed_points.add(sta)
    for sta, data in soft_constraints.items():
        coords[sta].append(tuple(map(float, data["value"])))
        seed_points.add(sta)

    # If Free Net Adjustments, pick a reference point and place it at the origin
    if not seed_points:
        if reference is None:
            # Pick the first station from the first baseline as a default reference
            reference = baselines[0].from_station if baselines else next(iter(params.keys()))

        origin = (0.0,) * dim
        coords[reference].append(origin)
        seed_points.add(reference)

    # 3. Build adjacency graph for traversal
    graph = defaultdict(list)
    for b in baselines:
        vec = np.asarray(b.vector, dtype=float)
        graph[b.from_station].append((b.to_station, vec))
        graph[b.to_station].append((b.from_station, -vec))

    # 4. Propagate coordinates via Breadth-First Search (BFS)
    queue = deque(seed_points)
    visited = set(seed_points)

    while queue:
        current_station = queue.popleft()
        if not coords[current_station]: continue  # Skip if a station has no coordinates to propagate from

        base_coord = np.mean(coords[current_station], axis=0)  # Use the average of known guesses for propagation

        for neighbor, vector in graph.get(current_station, []):
            # Calculate the neighbor's coordinate from the current station
            new_guess = tuple(base_coord + vector)
            coords[neighbor].append(new_guess)

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # 5. Average all collected guesses to get final initial coordinates
    final_coords = {}
    for sta, guesses in coords.items():
        if guesses:
            final_coords[sta] = tuple(np.mean(guesses, axis=0))

    # Re-apply soft constraints to ensure they override any propagated/averaged values
    for sta, data in soft_constraints.items():
        final_coords[sta] = tuple(map(float, data["value"]))

    # 6. Format the output for the adjustment engine
    values = {}  # Dict mapping symbols to float values
    X_list = []  # List of float values for unknown parameters
    new_params = {}  # The filtered dict of unknown parameters
    constants = {}  # Dict mapping symbols of fixed stations to float values

    # Correctly handle unpacking for 2D and 3D
    for sta, syms in params.items():
        if sta in hard_constraints:
            # This is a fixed station, add it to constants
            xyz = hard_constraints[sta]
            if dim == 2:
                X_sym, Y_sym = syms
                constants[X_sym] = float(xyz[0])
                constants[Y_sym] = float(xyz[1])
            else:  # 3D
                X_sym, Y_sym, Z_sym = syms
                constants[X_sym] = float(xyz[0])
                constants[Y_sym] = float(xyz[1])
                constants[Z_sym] = float(xyz[2])
            continue

        # This is an unknown station, add it to the initial guess vector
        new_params[sta] = syms

        # Get coordinates from our propagation, or default to origin if not found
        sta_coords = final_coords.get(sta, (0.0,) * dim)
        X_list.extend(sta_coords)

        if dim == 2:
            X_sym, Y_sym = syms
            values[X_sym] = sta_coords[0]
            values[Y_sym] = sta_coords[1]
        else:  # 3D
            X_sym, Y_sym, Z_sym = syms
            values[X_sym] = sta_coords[0]
            values[Y_sym] = sta_coords[1]
            values[Z_sym] = sta_coords[2]

    X_hat = np.array(X_list, dtype=float).reshape(-1, 1)

    return values, X_hat, constants, new_params

#
# ### Example Usage
# if __name__ == '__main__':
#     # A simple mock baseline class for the example
#     class Baseline:
#         def __init__(self, id, f, t, v):
#             self.baseline_id, self.from_station, self.to_station, self.vector = id, f, t, v
#
#
#     # --- 2D TEST ---
#     print("--- Running 2D Test ---")
#     st.session_state.dimension = '2D'
#
#     # Mock baselines and sympy parameters for a square network
#     baselines_2d = [
#         Baseline('B1', 'A', 'B', [100.1, 0.1]),
#         Baseline('B2', 'B', 'C', [-0.2, 99.8]),
#         Baseline('B3', 'A', 'D', [0.2, 100.2]),  # Redundant baseline
#     ]
#     params_2d = {
#         'A': sympy.symbols('X_A Y_A'),
#         'B': sympy.symbols('X_B Y_B'),
#         'C': sympy.symbols('X_C Y_C'),
#         'D': sympy.symbols('X_D Y_D'),
#     }
#     hard_constraints_2d = {'A': [5000.0, 5000.0]}
#
#     vals, xhat, consts, new_p = initial_guess(
#         baselines=baselines_2d,
#         params=params_2d,
#         hard_constraints=hard_constraints_2d
#     )
#
#     print("Initial Values (symbols):", vals)
#     print("Initial Guess Vector (X_hat):\n", xhat)
#     print("Constants (fixed stations):", consts)
#     print("-" * 25)
#
#     # --- 3D TEST ---
#     print("\n--- Running 3D Test ---")
#     st.session_state.dimension = '3D'
#
#     baselines_3d = [Baseline('B1', 'P1', 'P2', [10.0, 20.0, 5.0])]
#     params_3d = {
#         'P1': sympy.symbols('X_1 Y_1 Z_1'),
#         'P2': sympy.symbols('X_2 Y_2 Z_2'),
#     }
#
#     # Free Net Adjustments, will use 'P1' as reference at origin (0,0,0)
#     vals, xhat, consts, new_p = initial_guess(
#         baselines=baselines_3d,
#         params=params_3d,
#     )
#
#     print("Initial Values (symbols):", vals)
#     print("Initial Guess Vector (X_hat):\n", xhat)
#     print("Constants (fixed stations):", consts)