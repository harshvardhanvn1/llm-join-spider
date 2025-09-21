from __future__ import annotations
from typing import Dict
from joinbench.data.sqlite_utils import load_column_values_resolved

def predict_pair(db_path: str, left_table: str, left_col: str,
                 right_table: str, right_col: str, threshold: float = 0.05) -> Dict:
    """
    Return {"label": 0/1, "score": float, "explain": str}.
    Score = min(|A∩B|/|A|, |A∩B|/|B|) using DISTINCT normalized values.
    """
    A = load_column_values_resolved(db_path, left_table, left_col)
    B = load_column_values_resolved(db_path, right_table, right_col)
    if not A or not B:
        return {"label": 0, "score": 0.0, "explain": "no values"}
    inter = len(A & B)
    ca = inter / len(A) if A else 0.0
    cb = inter / len(B) if B else 0.0
    c = min(ca, cb)
    return {
        "label": 1 if c >= threshold else 0,
        "score": float(c),
        "explain": f"C={c:.3f} (|∩|/|A|={ca:.3f}, |∩|/|B|={cb:.3f})"
    }
