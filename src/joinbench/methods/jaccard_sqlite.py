from __future__ import annotations
from typing import Dict, Set
import re

from joinbench.data.sqlite_utils import load_column_values_resolved

# ---------------- helpers: normalization, tokens, acronym ----------------

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

def _tokens(s: str) -> Set[str]:
    return set(m.group(0).lower() for m in _token_re.finditer(s))

def _acronym(name: str) -> str:
    """publication -> p, author -> a, takes_classes -> tc, 'Domain Author' -> da"""
    toks = [t for t in _tokens(name) if t]
    return "".join(t[0] for t in toks) if toks else ""

# ---------------- name & heuristic similarity ----------------

def _name_similarity(lt: str, lc: str, rt: str, rc: str) -> float:
    # column token Jaccard (heavy), table token Jaccard (light)
    ctok_l, ctok_r = _tokens(lc), _tokens(rc)
    ttok_l, ttok_r = _tokens(lt), _tokens(rt)

    def jacc(a, b):
        if not a and not b:
            return 0.0
        inter = len(a & b)
        uni = len(a | b)
        return inter / uni if uni else 0.0

    col_sim = jacc(ctok_l, ctok_r)
    tab_sim = jacc(ttok_l, ttok_r)

    # strong exact/normalized equality bonus
    exact = 1.0 if lc.lower() == rc.lower() or _norm_ident(lc) == _norm_ident(rc) else 0.0

    # weighted blend: prefer column-name agreement
    base = max(exact, 0.85 * col_sim + 0.15 * tab_sim)

    # --- acronym-ID heuristic ---
    # If right column looks like <acronym(right_table)> + 'id' (e.g., 'pid', 'aid', 'cid', 'jid'),
    # and left column looks like a referencing slot (common FK words or mentions the right table),
    # give a conservative boost. This catches Spider pairs like cite.citing -> publication.pid.
    acr = _acronym(rt) + "id"
    looks_like_id_col = (rc.lower() == "id" or rc.lower() == acr)
    left_ref_tokens = {"ref", "refid", "fk", "foreign", "parent", "child", "source", "target", "cited", "citing"}
    left_is_refish = bool(_tokens(lc) & left_ref_tokens) or (_tokens(rt) & _tokens(lc))

    heuristic = 0.0
    if looks_like_id_col and left_is_refish:
        # push toward a high-confidence match, but not absolute 1.0
        heuristic = 0.95

    return max(base, heuristic)

# ---------------- main API ----------------

def predict_pair(db_path: str, left_table: str, left_col: str,
                 right_table: str, right_col: str, threshold: float = 0.05) -> Dict:
    """
    Primary: Jaccard(A,B) on DISTINCT normalized values.
    Fallback: when either side is too sparse (|A|<3 or |B|<3),
              use a conservative name/heuristic score that recognizes
              acronym-ID patterns (e.g., publication.pid) and reference-like
              columns (e.g., citing/cited).
    """
    A = load_column_values_resolved(db_path, left_table, left_col)
    B = load_column_values_resolved(db_path, right_table, right_col)

    # If both sides have enough values, use value Jaccard
    if len(A) >= 3 and len(B) >= 3:
        inter = len(A & B)
        uni = len(A | B)
        j = inter / uni if uni else 0.0
        return {
            "label": 1 if j >= threshold else 0,
            "score": float(j),
            "explain": f"J={j:.3f} (|∩|={inter}, |∪|={uni})"
        }

    # Fallback: name/heuristic similarity (threshold tuned for precision)
    name_score = _name_similarity(left_table, left_col, right_table, right_col)
    name_threshold = 0.90
    return {
        "label": 1 if name_score >= name_threshold else 0,
        "score": float(name_score),
        "explain": f"name-fallback: {left_table}.{left_col} ~ {right_table}.{right_col} (score={name_score:.3f})"
    }
