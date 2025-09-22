from __future__ import annotations
import os, json, re
from typing import Dict, List, Tuple, Set, Optional
import sqlite3

from joinbench.data.sqlite_utils import (
    resolve_table_name, resolve_column_name,
    load_column_values_sqlite, quote_ident,
)

# Optional: load .env if present
try:
    import dotenv; dotenv.load_dotenv()
except Exception:
    pass

# ------------- helpers -------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("empty response")
    m = _JSON_RE.search(text)
    if m:
        return m.group(0)
    # fallback brace counter
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

def _load_small_samples(db_path: str, table: str, column: str, k: int = 8):
    vals: Set[str] = load_column_values_sqlite(db_path, table, column, sample_limit=50_000)
    vs = sorted(vals)[:k]
    return vs, len(vals)

def _table_cols_preview(db_path: str, table: str, k: int = 8) -> List[str]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    try:
        cur.execute(f'PRAGMA table_info({quote_ident(table)})')
        cols = [r[1] for r in cur.fetchall()]
    finally:
        con.close()
    return cols[:k]

def _default_prompt(db_id: str,
                    left: Tuple[str, str, List[str], int],
                    right: Tuple[str, str, List[str], int],
                    left_table_cols: List[str],
                    right_table_cols: List[str]) -> str:
    lt, lc, lvals, lcard = left
    rt, rc, rvals, rcard = right
    return f"""You are a careful database assistant. Decide if two SQL columns are likely an equality join key.

Return ONLY a compact JSON with fields:
  "label": 1 or 0,
  "score": number in [0,1] (confidence they should be joined by equality),
  "reason": brief string (<=200 chars).

RULES:
- Consider that data may be sparse or empty.
- Prefer precision over recall; only output 1 when strong evidence exists (name match, semantics, samples, or schema hints).
- Do NOT invent data values. Base your judgment on provided names and samples only.
- Output VALID JSON only. No markdown. No extra text.

Database: {db_id}

LEFT  table.column: {lt}.{lc}
LEFT  table columns: {left_table_cols}
LEFT  distinct sample (size={lcard}): {lvals}

RIGHT table.column: {rt}.{rc}
RIGHT table columns: {right_table_cols}
RIGHT distinct sample (size={rcard}): {rvals}

Now output the JSON:
{{"label": 0, "score": 0.0, "reason": ""}}
""".strip()

# ------------- Gemini call -------------

def _call_gemini(prompt: str, model: Optional[str] = None, generation_config: Optional[dict] = None) -> str:
    # We ignore per-call model/config here to keep one shared limiter.
    from joinbench.llm.gemini_client import get_client
    client = get_client()
    return client.generate_text(prompt)


def _parse_llm_json(text: str) -> Dict:
    block = _extract_json_block(text)
    obj = json.loads(block)
    if "label" not in obj or "score" not in obj:
        raise ValueError("JSON missing required keys")
    obj["label"] = 1 if int(obj["label"]) == 1 else 0
    obj["score"] = float(obj["score"])
    obj["reason"] = str(obj.get("reason", ""))
    obj["score"] = max(0.0, min(1.0, obj["score"]))
    return obj

# ------------- public API -------------

def predict_pair(db_path: str, left_table: str, left_col: str,
                 right_table: str, right_col: str,
                 threshold: float = 0.5,
                 model: Optional[str] = None,
                 sample_k: int = 8) -> Dict:
    """
    Gemini-based join decision with strict JSON output and no hallucinated data.
    """
    # resolve identifiers
    lt = resolve_table_name(db_path, left_table)
    lc = resolve_column_name(db_path, lt, left_col)
    rt = resolve_table_name(db_path, right_table)
    rc = resolve_column_name(db_path, rt, right_col)

    # small samples + table previews
    lvals, lcard = _load_small_samples(db_path, lt, lc, k=sample_k)
    rvals, rcard = _load_small_samples(db_path, rt, rc, k=sample_k)
    lcols = _table_cols_preview(db_path, lt)
    rcols = _table_cols_preview(db_path, rt)

    prompt = _default_prompt(
        db_id=os.path.basename(db_path).replace(".sqlite", ""),
        left=(lt, lc, lvals, lcard),
        right=(rt, rc, rvals, rcard),
        left_table_cols=lcols,
        right_table_cols=rcols,
    )

    # call Gemini with one retry if JSON parsing fails
    try:
        raw = _call_gemini(prompt, model=model)
        obj = _parse_llm_json(raw)
    except Exception:
        strict_prompt = prompt + "\n\nREMINDER: Output ONLY a valid JSON object. No extra text."
        raw = _call_gemini(strict_prompt, model=model)
        obj = _parse_llm_json(raw)

    label = 1 if (obj["label"] == 1 and obj["score"] >= threshold) else 0
    explain = f'LLM score={obj["score"]:.3f}, raw_label={obj["label"]}, reason="{obj.get("reason","")[:180]}"'
    return {
        "label": label,
        "score": float(obj["score"]),
        "explain": explain
    }
