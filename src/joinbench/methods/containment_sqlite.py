from __future__ import annotations
from typing import Dict
from joinbench.data.sqlite_utils import load_column_values_resolved

def predict_pair(db_path: str,
                 left_table: str, left_col: str,
                 right_table: str, right_col: str,
                 threshold: float = 0.05) -> dict:
    """
    Value-overlap Containment: |A ∩ B| / min(|A|, |B|) on DISTINCT non-null values.
    """
    from joinbench.data.sqlite_utils import load_column_values_resolved

    L_vals = load_column_values_resolved(db_path, left_table, left_col)
    R_vals = load_column_values_resolved(db_path, right_table, right_col)

    L = set(L_vals)
    R = set(R_vals)

    if not L or not R:
        score = 0.0
        explain = "no values"
    else:
        inter = len(L & R)
        denom = max(min(len(L), len(R)), 1)
        score = inter / denom
        explain = f"containment: |∩|={inter}, min(|L|,|R|)={denom}"

    label = 1 if score >= float(threshold) else 0
    return {
        "label": label,
        "score": float(score),
        "explain": explain
    }

