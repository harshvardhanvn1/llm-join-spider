from __future__ import annotations
import os, json, sqlite3, re, itertools
from typing import List, Dict, Tuple, Optional, Set

from joinbench.llm.gemini_client import get_client
from joinbench.data.sqlite_utils import quote_ident

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty response")
    m = _JSON_RE.search(text)
    if m:
        return m.group(0)
    s = text.strip()
    depth = 0; start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return s[start:i+1]
    raise ValueError("no JSON object found")

def _pragma_table_info(con: sqlite3.Connection, table: str):
    cur = con.cursor()
    cur.execute(f'PRAGMA table_info({quote_ident(table)})')
    rows = cur.fetchall()
    # cid, name, type, notnull, dflt_value, pk
    return [{"name": r[1], "type": (r[2] or "").upper(), "pk": (r[5] == 1)} for r in rows]

def _pragma_foreign_keys(con: sqlite3.Connection, table: str):
    cur = con.cursor()
    cur.execute(f'PRAGMA foreign_key_list({quote_ident(table)})')
    rows = cur.fetchall()
    # id,seq,ref_table,from_col,to_col,on_update,on_delete,match
    fks = []
    for (_id, _seq, ref_table, from_col, to_col, *_rest) in rows:
        fks.append({"from": from_col, "to_table": ref_table, "to_col": to_col})
    return fks

def _collect_schema(con: sqlite3.Connection, tables: List[str]) -> Dict[str, Dict]:
    schema = {}
    for t in tables:
        cols = _pragma_table_info(con, t)
        fks = _pragma_foreign_keys(con, t)
        schema[t] = {"columns": cols, "fks": fks}
    return schema

def _looks_like_id(name: str) -> bool:
    n = name.lower()
    return n == "id" or n.endswith("_id") or n.endswith("id")

def _build_candidates(schema: Dict[str, Dict]) -> List[Tuple[str,str,str,str]]:
    """
    Build a small, plausible set of candidate edges across the provided tables.
    Priority:
      1) Declared FKs
      2) pk <-> fk/id-like
      3) id-like <-> id-like across tables
      4) same-name columns across tables
    Capped to 25 to keep prompt short.
    """
    cands: List[Tuple[str,str,str,str]] = []
    tables = list(schema.keys())

    # 1) Declared FKs
    for lt, linfo in schema.items():
        for fk in linfo["fks"]:
            rt = fk["to_table"]
            if rt in schema:
                lc, rc = fk["from"], fk["to_col"]
                pair = _norm_edge((lt, lc, rt, rc))
                if pair not in cands:
                    cands.append(pair)

    # 2) pk <-> id-like across different tables
    for lt, rt in itertools.combinations(tables, 2):
        lcols = schema[lt]["columns"]; rcols = schema[rt]["columns"]
        l_pks = [c["name"] for c in lcols if c["pk"]]
        r_pks = [c["name"] for c in rcols if c["pk"]]
        l_ids = [c["name"] for c in lcols if _looks_like_id(c["name"])]
        r_ids = [c["name"] for c in rcols if _looks_like_id(c["name"])]

        for lc in l_pks or l_ids:
            for rc in r_ids or r_pks:
                pair = _norm_edge((lt, lc, rt, rc))
                if pair not in cands:
                    cands.append(pair)
        for rc in r_pks or r_ids:
            for lc in l_ids or l_pks:
                pair = _norm_edge((lt, lc, rt, rc))
                if pair not in cands:
                    cands.append(pair)

    # 3) id-like <-> id-like (fallback)
    for lt, rt in itertools.combinations(tables, 2):
        for lc in [c["name"] for c in schema[lt]["columns"] if _looks_like_id(c["name"])]:
            for rc in [c["name"] for c in schema[rt]["columns"] if _looks_like_id(c["name"])]:
                pair = _norm_edge((lt, lc, rt, rc))
                if pair not in cands:
                    cands.append(pair)

    # 4) same-name columns across tables
    for lt, rt in itertools.combinations(tables, 2):
        lnames = {c["name"].lower() for c in schema[lt]["columns"]}
        rnames = {c["name"].lower() for c in schema[rt]["columns"]}
        for name in sorted(lnames & rnames):
            pair = _norm_edge((lt, name, rt, name))
            if pair not in cands:
                cands.append(pair)

    # cap
    return cands[:25]

def _norm_edge(edge: Tuple[str,str,str,str]) -> Tuple[str,str,str,str]:
    lt, lc, rt, rc = edge
    lt, lc, rt, rc = str(lt), str(lc), str(rt), str(rc)
    if (rt, rc) < (lt, lc):
        lt, lc, rt, rc = rt, rc, lt, lc
    return (lt, lc, rt, rc)

def _prompt_candidates(db_id: str, question: str, tables: List[str], candidates: List[Tuple[str,str,str,str]]) -> str:
    enum = [
        {"i": i, "left_table": lt, "left_column": lc, "right_table": rt, "right_column": rc}
        for i, (lt, lc, rt, rc) in enumerate(candidates)
    ]
    enum_text = json.dumps(enum, ensure_ascii=False)
    return f"""You are a careful database assistant. Given a natural language question and a list of plausible join candidates (by index), pick only the join edges that are necessary to answer the question.

Return ONLY valid JSON:
  "chosen": [indices of selected candidates],
  "reason": brief string (<=200 chars).

Rules:
- Choose from the provided candidates ONLY. Do not invent other joins.
- Prefer precision over recall; if no joins are needed, return an empty list.
- Output VALID JSON only. No markdown.

Database: {db_id}
Question: {question}

Candidates (JSON array, each has an 'i' index):
{enum_text}

Now output the JSON:
{{"chosen": [], "reason": ""}}
""".strip()

def _parse_choice_json(text: str) -> Dict:
    obj = json.loads(_extract_json_block(text))
    chosen = obj.get("chosen", [])
    if not isinstance(chosen, list):
        chosen = []
    chosen = [int(i) for i in chosen if isinstance(i, (int, float, str)) and str(i).lstrip("-").isdigit()]
    obj["chosen"] = [i for i in chosen if i >= 0]
    obj["reason"] = str(obj.get("reason", ""))[:200]
    return obj

def predict_joins(db_path: str, db_id: str, question: str, tables_in_query: List[str]) -> Dict:
    """
    Candidate-selection LLM:
      1) Build candidate edges from schema signals (PK/FK/id-like/same-name).
      2) Ask the model to select by indices.
      3) Validate & return normalized edges.
    """
    con = sqlite3.connect(db_path)
    try:
        schema = _collect_schema(con, tables_in_query)
    finally:
        con.close()

    candidates = _build_candidates(schema)

    # If no candidates, trivially no joins
    if not candidates:
        return {"pred_joins": [], "explain": "No plausible join candidates for the tables in query."}

    prompt = _prompt_candidates(db_id, question, tables_in_query, candidates)
    client = get_client()
    try:
        raw = client.generate_text(prompt)
        obj = _parse_choice_json(raw)
    except Exception:
        raw = client.generate_text(prompt + "\n\nREMINDER: Output ONLY a valid JSON object with 'chosen' and 'reason'.")
        obj = _parse_choice_json(raw)

    # map indices -> edges; validate bounds
    chosen_edges: List[Tuple[str,str,str,str]] = []
    for i in obj["chosen"]:
        if 0 <= i < len(candidates):
            chosen_edges.append(_norm_edge(candidates[i]))

    # dedupe & pack
    chosen_edges = sorted(set(chosen_edges))
    pred = [{"left_table": lt, "left_column": lc, "right_table": rt, "right_column": rc}
            for (lt, lc, rt, rc) in chosen_edges]
    return {"pred_joins": pred, "explain": obj.get("reason", "")}
