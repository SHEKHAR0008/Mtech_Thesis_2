import numpy as np
import networkx as nx
from typing import List, Tuple, Dict


# # Mocking Streamlit's session_state for a runnable standalone example
# # In your actual Streamlit app, you would remove this class.
# class MockSessionState:
#     """A mock class to simulate st.session_state for testing."""
#
#     def __init__(self, dimension='3D'):
#         self.dimension = dimension
#
#
# # You would use 'st.session_state' in your app. We use this mock object for the example.
# st_session_state = MockSessionState()


class Baseline:
    """Represents a single baseline vector between two stations."""

    def __init__(self, baseline_id: str, from_station: str, to_station: str, vector: List[float],
                 vcv: List[List[float]] = None):
        self.baseline_id = baseline_id
        self.from_station = from_station
        self.to_station = to_station
        self.vector = np.array(vector, dtype=float)
        self.vcv = np.array(vcv, dtype=float) if vcv is not None else None

    def __repr__(self):
        return f"{self.baseline_id}: {self.from_station}->{self.to_station}|{self.vector}"


def find_loops(baselines: List[Baseline]) -> List[List[Tuple[Baseline, int]]]:
    """
    Finds all fundamental cycles (loops) in the network of baselines.
    This function is dimension-agnostic as it only considers the station connections.
    """
    # Build multimap for all connections to get directionality
    edge_map = {}
    for b in baselines:
        edge_map[(b.from_station, b.to_station)] = (b, 1)  # Forward direction
        edge_map[(b.to_station, b.from_station)] = (b, -1)  # Reverse direction

    # Build a simple undirected graph to find the cycles
    G = nx.Graph()
    for b in baselines:
        G.add_edge(b.from_station, b.to_station)

    # Get the fundamental cycles from the graph
    cycles = nx.cycle_basis(G)

    loops = []
    for cycle in cycles:
        n = len(cycle)
        loop = []
        for i in range(n):
            start_node = cycle[i]
            end_node = cycle[(i + 1) % n]

            # Find the corresponding baseline and its sign (direction)
            if (start_node, end_node) in edge_map:
                baseline, sign = edge_map[(start_node, end_node)]
                loop.append((baseline, sign))
            else:
                # This should not happen in a valid cycle from networkx
                break

        if len(loop) == n:  # Ensure the loop was fully constructed
            loops.append(loop)

    return loops


def check_all_loops(baselines: List[Baseline], threshold: float, dimension: str) -> Tuple[bool, str, List[Dict]]:
    """
    Calculates the misclosure for all loops in the network.
    This function is now dimension-aware.

    Args:
        baselines: A list of Baseline objects.
        threshold: The maximum acceptable misclosure magnitude.
        dimension: The expected dimension of the data ('2D' or '3D').

    Returns:
        A tuple containing: (overall_ok, summary_message, details_list).
    """
    if dimension == '2D':
        dim = 2
    elif dimension == '3D':
        dim = 3
    else:
        # As requested, 1D is not implemented yet.
        raise NotImplementedError("Dimension must be '2D' or '3D'. 1D support is not implemented.")

    # --- Data Validation ---
    # Check if all baseline vectors match the specified dimension
    for b in baselines:
        if len(b.vector) != dim:
            raise ValueError(
                f"Mismatched dimensions: Expected {dimension} ({dim} elements), "
                f"but baseline '{b.baseline_id}' has a vector with {len(b.vector)} elements."
            )

    loops = find_loops(baselines)
    if not loops:
        return True, "No loops found in the network to check.", []

    details = []
    any_fail = False
    messages = []

    for i, loop in enumerate(loops):
        # Initialize the misclosure vector based on the correct dimension
        misclosure_vector = np.zeros(dim)

        baseline_ids_in_loop = []
        station_path = [loop[0][0].from_station if loop[0][1] == 1 else loop[0][0].to_station]

        for (baseline, sign) in loop:
            misclosure_vector += sign * baseline.vector

            # For logging and reporting
            baseline_ids_in_loop.append(f"{baseline.baseline_id}" if sign == 1 else f"-{baseline.baseline_id}")
            next_station = baseline.to_station if sign == 1 else baseline.from_station
            station_path.append(next_station)

        misclosure_magnitude = np.linalg.norm(misclosure_vector)
        is_ok = misclosure_magnitude <= threshold

        if not is_ok:
            any_fail = True

        details.append({
            "loop_index": i,
            "baseline_ids": baseline_ids_in_loop,
            "stations": station_path,
            "misclosure_vector": misclosure_vector.tolist(),
            "magnitude": misclosure_magnitude,
            "ok": is_ok
        })

        result_message = (
            f"--- Loop {i + 1} ---\n"
            f"Path: {' -> '.join(station_path)}\n"
            f"Baselines Used: {', '.join(baseline_ids_in_loop)}\n"
            f"Misclosure Vector: {np.round(misclosure_vector, 5)}\n"
            f"Misclosure Magnitude: {misclosure_magnitude:.5f} m\n"
            f"Threshold: {threshold} m\n"
            f"Result: {'✅ OK' if is_ok else '❌ FAIL'}\n"
        )
        messages.append(result_message)

    summary = '\n'.join(messages)
    return not any_fail, summary, details

#
# ### Example Usage
# if __name__ == "__main__":
#
#     print("====================")
#     print("--- 2D DATA TEST ---")
#     print("====================")
#
#     # Set the session state for 2D
#     st_session_state.dimension = '2D'
#     print(f"Running with dimension set to: '{st_session_state.dimension}'\n")
#
#     # A simple 2D triangular loop: A -> B -> C -> A
#     # To close the loop, AB + BC + CA should equal [0, 0]
#     # Let's introduce a small error in baseline 'CA' to see a failure.
#     b_2d_1 = Baseline('AB', 'A', 'B', [10.0, 5.0])
#     b_2d_2 = Baseline('BC', 'B', 'C', [-4.0, 8.0])
#     b_2d_3 = Baseline('CA', 'C', 'A', [-6.05, -13.0])  # True value would be [-6.0, -13.0]
#
#     baselines_2d = [b_2d_1, b_2d_2, b_2d_3]
#
#     # Note: We pass the dimension from our session state to the function
#     ok_2d, msg_2d, det_2d = check_all_loops(baselines_2d, threshold=0.01, dimension=st_session_state.dimension)
#
#     print(msg_2d)
#     if not ok_2d:
#         print("DETAILS OF FAILED LOOP(S):")
#         for detail in det_2d:
#             if not detail['ok']:
#                 print(detail)
#
#     print("\n====================")
#     print("--- 3D DATA TEST ---")
#     print("====================")
#
#     # Set the session state for 3D
#     st_session_state.dimension = '3D'
#     print(f"Running with dimension set to: '{st_session_state.dimension}'\n")
#
#     # Original 3D triangular loop: A -> B -> C -> A
#     b_3d_1 = Baseline('AB', 'A', 'B', [10.0, 5.0, 2.0])
#     b_3d_2 = Baseline('BC', 'B', 'C', [-4.0, 8.0, 1.0])
#     b_3d_3 = Baseline('AC', 'A', 'C', [6.0, 13.0, 3.0])  # Note this is AC, not CA
#
#     baselines_3d = [b_3d_1, b_3d_2, b_3d_3]
#
#     # The loop will be A -> B -> C -> A. This means AB + BC - AC should be near zero.
#     ok_3d, msg_3d, det_3d = check_all_loops(baselines_3d, threshold=0.01, dimension=st_session_state.dimension)
#     print(msg_3d)