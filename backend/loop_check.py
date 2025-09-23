import numpy as np
import networkx as nx
from typing import List, Tuple, Dict

class Baseline:
    def __init__(self, baseline_id, from_station, to_station, vector, vcv=None):
        self.baseline_id = baseline_id
        self.from_station = from_station
        self.to_station = to_station
        self.vector = np.array(vector, dtype=float)
        self.vcv = np.array(vcv, dtype=float) if vcv is not None else None

    def __repr__(self):
        return f"{self.baseline_id}: {self.from_station}->{self.to_station}|{self.vector}"

def find_loops(baselines: List[Baseline]) -> List[List[Tuple[Baseline, int]]]:
    # Build multimap for all connections
    edge_map = {}
    for b in baselines:
        edge_map[(b.from_station, b.to_station)] = (b, 1)
        edge_map[(b.to_station, b.from_station)] = (b, -1)
    # Build plain undirected graph for cycle finding
    G = nx.Graph()
    for b in baselines:
        G.add_edge(b.from_station, b.to_station)
    cycles = nx.cycle_basis(G)
    loops = []
    for cycle in cycles:
        n = len(cycle)
        loop = []
        for i in range(n):
            start = cycle[i]
            end = cycle[(i+1) % n]
            if (start, end) in edge_map:
                b, sign = edge_map[(start, end)]
                loop.append((b, sign))
            else:
                # No baseline found in any direction between these two - error for this loop
                break
        if len(loop) == n:  # Only add if complete
            loops.append(loop)
    return loops

def check_all_loops(baselines, threshold: float):
    loops = find_loops(baselines)
    details = []
    any_fail = False
    messages = []

    for loop in loops:
        total = np.zeros(3)
        ids = []
        stations = []
        sign_list = []
        for idx, (b, sign) in enumerate(loop):
            total += sign * b.vector
            ids.append((b.baseline_id if sign == 1 else f"-{b.baseline_id}"))
            stations.append(b.from_station if sign == 1 else b.to_station)
            sign_list.append(sign)
        last_station = loop[-1][0].to_station if loop[-1][1] == 1 else loop[-1][0].from_station
        stations.append(last_station)
        mag = np.linalg.norm(total)
        ok = mag <= threshold
        details.append({
            "loop_ids": ids,
            "stations": stations,
            "signs": sign_list,
            "misclosure_vector": total.copy(),
            "magnitude": mag,
            "ok": ok
        })
        result = (
            f"Loop: {' -> '.join(stations)}"
            f"\nBaselines: {ids}"
            f"\nMisclosure: {mag:.5f} m "
            f"(threshold: {threshold} m) "
            f"{'OK' if ok else 'FAIL'}"
            f"\nVector: {total}\n"
        )
        messages.append(result)
        if not ok:
            any_fail = True

    summary = '\n'.join(messages)
    return not any_fail, summary, details


# # Example usage
# if __name__ == "__main__":
#     # Baselines: AB, AC, CB; loop is A-B-C-A (needs AC and CB to be walked in reverse if starting from A-B)
#     b1 = Baseline('AB', 'A', 'B', [1,0,0])
#     b2 = Baseline('AC', 'A', 'C', [2,0,0])
#     b3 = Baseline('CB', 'C', 'B', [-1,0,0])
#     test = [b1, b2, b3]
#     ok, msg, det = check_all_loops(test, threshold=0.01)
#     print(msg)
#     if not ok:
#         print(det)
