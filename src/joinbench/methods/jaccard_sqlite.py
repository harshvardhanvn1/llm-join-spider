from __future__ import annotations
from typing import Dict
from joinbench.data.sqlite_utils import load_column_values_resolved

def predict_pair(db_path: str, left_table: str, left_col: str,
                 right_table: str, right_col: str, threshold: float = 0.05) -> Dict:
    """
    Return {"label": 0/1, "score": float, "explain": str}.
    Score = Jaccard(A,B) over DISTINCT normalized values.
    """
    A = load_column_values_resolved(db_path, left_table, left_col)
    B = load_column_values_resolved(db_path, right_table, right_col)
    if not A or not B:
        return {"label": 0, "score": 0.0, "explain": "no values"}
    inter = len(A & B)
    uni = len(A | B)
    j = inter / uni if uni else 0.0
    return {
        "label": 1 if j >= threshold else 0,
        "score": float(j),
        "explain": f"J={j:.3f} (|∩|={inter}, |∪|={uni})"
    }
