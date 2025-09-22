from __future__ import annotations
from typing import Dict, Set
import re
from joinbench.data.sqlite_utils import load_column_values_resolved

def _norm_ident(s: str) -> str:
    out = []
    last_us = False
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch); last_us = False
        else:
            if not last_us:
                out.append("_"); last_us = True
    while out and out[0] == "_": out.pop(0)
    while out and out[-1] == "_": out.pop()
    return "".join(out)

_token_re = re.compile(r"[A-Za-z0-9]+")

def _acronym(name: str) -> str:
    toks = [m.group(0).lower() for m in _token_re.finditer(name)]
    return "".join(t[0] for t in toks) if toks else ""

def _strict_name_fallback(left_table: str, left_col: str,
                          right_table: str, right_col: str) -> float:
    """
    STRICT fallback:
    - pass only if column names are exactly equal (case-insensitive) OR
      normalized-equal (underscores/spaces/case differences ignored)
    - OR both are exactly the same acronym+id (e.g., pid vs pid, aid vs aid)
    Returns a score in [0,1]; we’ll treat >= 1.0 as positive.
    """
    lc, rc = left_col, right_col
    if lc.lower() == rc.lower():
        return 1.0
    if _norm_ident(lc) == _norm_ident(rc):
        return 1.0

    # strict acronym-id: e.g., publication.pid -> pid ; author.aid -> aid
    aid_l = lc.lower()
    aid_r = rc.lower()
    # allow either side to be exactly acronym+id of its own table
    acr_l = _acronym(left_table) + "id" if left_table else ""
    acr_r = _acronym(right_table) + "id" if right_table else ""
    if aid_l == aid_r and (aid_l == acr_l or aid_r == acr_r):
        return 1.0

    return 0.0

def predict_pair(db_path: str,
                 left_table: str, left_col: str,
                 right_table: str, right_col: str,
                 threshold: float = 0.05) -> dict:
    """
    Value-overlap Jaccard on DISTINCT non-null cell values.
    Falls back to score=0.0 when either side has no values.
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
        union = len(L | R) or 1
        score = inter / union
        explain = f"overlap: |∩|={inter}, |∪|={union}"

    label = 1 if score >= float(threshold) else 0
    return {
        "label": label,
        "score": float(score),
        "explain": explain
    }

