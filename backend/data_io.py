import numpy as np
import re
import csv
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

def parse_baseline_text(text: str) -> (List[Baseline],List):
    """
    Parse user-pasted/uploaded baseline input (TXT/CSV format).
    Returns a list of Baseline objects.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.lower().startswith("from")]
    delimiter = ',' if lines and ',' in lines[0] else None
    if delimiter:
        parsed = list(csv.reader(lines))
    else:
        parsed = [re.split(r'\s+', l) for l in lines]
    output = []
    unique_stations = set()  # temporary set to collect stations

    if st.session_state.dimension == '2D':
        rows_per_baseline = 2
        # Expected column indices for 2D data
        vec_cols = [3, 4]
        cov_cols = [[4, 5], [5, 6]]
    else: # Default 3D
        rows_per_baseline = 3
        # Expected column indices for 3D data
        vec_cols = [3, 3, 3]
        cov_cols = [[4, 0, 0], [4, 5, 0], [4, 5, 6]]

    i = 0
    while i < len(parsed):
        rows = parsed[i:i + rows_per_baseline]
        if len(rows) < rows_per_baseline:
            break
        from_sta, to_sta = rows[0][0], rows[0][1]
        unique_stations.update([from_sta, to_sta])

        if st.session_state.dimension == '2D':
            vec = [float(rows[0][3]), float(rows[1][3])]
            cov = [[0, 0], [0, 0]]
            cov[0][0] = float(rows[0][4])
            cov[1][0], cov[1][1] = float(rows[1][4]), float(rows[1][5])
            cov[0][1] = cov[1][0]
        else:
            vec = [float(rows[0][3]), float(rows[1][3]), float(rows[2][3])]
            cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            cov[0][0] = float(rows[0][4])
            cov[1][0], cov[1][1] = float(rows[1][4]), float(rows[1][5])
            cov[2][0], cov[2][1], cov[2][2] = float(rows[2][4]), float(rows[2][5]), float(rows[2][6])
            cov[0][1], cov[0][2], cov[1][2] = cov[1][0], cov[2][0], cov[2][1]

        baseline_id = f"{from_sta}_{to_sta}"
        output.append(Baseline(baseline_id, from_sta, to_sta, vec, cov))
        i += rows_per_baseline

    station_list = list(unique_stations)
    return output,station_list