from __future__ import annotations
from typing import Set, List
import sqlite3

# ---------- Identifier handling ----------

def _normalize_ident(s: str) -> str:
    """
    Normalize an identifier for loose matching:
    - lowercase
    - non-alnum → underscore
    - collapse repeats; trim underscores
    """
    out = []
    last_us = False
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
            last_us = False
        else:
            if not last_us:
                out.append("_")
            last_us = True
    # trim underscores at ends
    while out and out[0] == "_": out.pop(0)
    while out and out[-1] == "_": out.pop()
    return "".join(out)

def quote_ident(name: str) -> str:
    """Double-quote an identifier for SQLite (escape internal quotes)."""
    return '"' + name.replace('"', '""') + '"'

# ---------- Introspection ----------

def list_tables(db_path: str) -> List[str]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()

def list_columns(db_path: str, table: str) -> List[str]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(f'PRAGMA table_info({quote_ident(table)})')
        return [r[1] for r in cur.fetchall()]
    finally:
        con.close()

# ---------- Resolvers (schema → actual DB identifiers) ----------

def resolve_table_name(db_path: str, desired: str) -> str:
    tables = list_tables(db_path)
    if desired in tables:
        return desired
    nd = _normalize_ident(desired)
    best, best_score = None, -1
    for t in tables:
        score = 0
        if t.lower() == desired.lower():
            score = 3
        elif _normalize_ident(t) == nd:
            score = 2
        elif nd in _normalize_ident(t) or _normalize_ident(t) in nd:
            score = 1
        if score > best_score:
            best, best_score = t, score
    return best if best is not None else desired

def resolve_column_name(db_path: str, table: str, desired: str) -> str:
    cols = list_columns(db_path, table)
    if desired in cols:
        return desired
    nd = _normalize_ident(desired)
    best, best_score = None, -1
    for c in cols:
        score = 0
        if c.lower() == desired.lower():
            score = 3
        elif _normalize_ident(c) == nd:
            score = 2
        elif nd in _normalize_ident(c) or _normalize_ident(c) in nd:
            score = 1
        if score > best_score:
            best, best_score = c, score
    return best if best is not None else desired

# ---------- Value loader (DISTINCT) ----------

def _normalize_value(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    return s if s != "" else None

def load_column_values_sqlite(db_path: str, table: str, column: str, sample_limit: int = 0) -> List[str]:
    """
    Return DISTINCT non-null values from table.column.
    Robust to non-UTF8 bytes stored in TEXT: fetch as bytes, then decode per value.
    """
    con = sqlite3.connect(db_path)
    try:
        # Fetch raw bytes so we control decoding
        con.text_factory = bytes
        cur = con.cursor()
        q = f"""SELECT DISTINCT {quote_ident(column)} FROM {quote_ident(table)} WHERE {quote_ident(column)} IS NOT NULL"""
        if sample_limit and sample_limit > 0:
            q += f" LIMIT {int(sample_limit)}"
        cur.execute(q)
        rows = cur.fetchall()

        out: List[str] = []
        for (val,) in rows:
            # val can be bytes, str, int, float, etc.
            if isinstance(val, bytes):
                try:
                    s = val.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        s = val.decode("latin-1")
                    except Exception:
                        s = val.decode("utf-8", errors="replace")
                out.append(s)
            elif isinstance(val, str):
                out.append(val)
            else:
                # Coerce non-strings to str
                out.append(str(val))
        return out
    finally:
        con.close()


def load_column_values_resolved(db_path: str, table_hint: str, column_hint: str, sample_limit: int = 50000) -> Set[str]:
    """
    Resolve (table_hint, column_hint) → actual DB identifiers, then load DISTINCT values.
    """
    t = resolve_table_name(db_path, table_hint)
    c = resolve_column_name(db_path, t, column_hint)
    return load_column_values_sqlite(db_path, t, c, sample_limit=sample_limit)
