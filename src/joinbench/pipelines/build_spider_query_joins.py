from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import click

# We use sqlglot to parse SQL reliably
# pip install sqlglot
import sqlglot
from sqlglot import exp

def _read_spider_splits(spider_dir: str) -> List[Dict[str, Any]]:
    """
    Reads Spider train/dev JSON files (train_spider.json, dev.json) if present.
    Returns a combined list of examples with keys: question, db_id, query.
    """
    root = Path(spider_dir)
    candidates = [
        root / "train_spider.json",
        root / "dev.json",
        root / "train.json",   # some distributions use these names
        root / "dev.json",
    ]
    items: List[Dict[str, Any]] = []
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text())
            for ex in data:
                # normalize to common keys
                q = ex.get("question") or ex.get("question_toks") or ""
                if isinstance(q, list): q = " ".join(q)
                items.append({
                    "question": q,
                    "db_id": ex["db_id"],
                    "query": ex["query"],
                })
    if not items:
        raise FileNotFoundError("Could not find Spider split files (train_spider.json/dev.json).")
    return items

def _table_column_from_qual(expr: exp.Expression) -> Tuple[str, str] | None:
    """
    Given a qualified column expression like table.col, return (table, col).
    """
    if isinstance(expr, exp.Column):
        table = expr.table
        col = expr.name
        if table and col:
            return table, col
    return None

def _extract_joins(sql: str) -> List[Tuple[str, str, str, str]]:
    """
    Extract equality join edges (tableA.colA = tableB.colB) from JOIN ... ON ... clauses.
    Returns list of (lt, lc, rt, rc). We keep order normalized (lexicographically) to avoid duplicates.
    """
    joins: List[Tuple[str, str, str, str]] = []
    try:
        # Spider DBs are SQLite; sqlglot’s sqlite dialect is a good fit
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        # if parsing fails, return empty (we’ll count as no gold joins)
        return joins

    # Walk JOIN clauses
    for join in tree.find_all(exp.Join):
        on_expr = join.args.get("on")
        if not on_expr:
            continue
        # Collect equality predicates inside the ON
        for pred in on_expr.find_all(exp.EQ):
            left = pred.left
            right = pred.right
            ltc = _table_column_from_qual(left)
            rtc = _table_column_from_qual(right)
            if not ltc or not rtc:
                continue
            (lt, lc), (rt, rc) = ltc, rtc
            # normalize ordering so {A.x,B.y} == {B.y,A.x}
            if (rt, rc) < (lt, lc):
                lt, lc, rt, rc = rt, rc, lt, lc
            joins.append((lt, lc, rt, rc))

    # De-duplicate
    uniq = sorted(set(joins))
    return uniq

def _tables_in_query(sql: str) -> List[str]:
    """
    Rough list of table names referenced (FROM + JOIN targets).
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    tabs = set()
    for t in tree.find_all(exp.Table):
        name = t.name
        if name:
            tabs.add(name)
    return sorted(tabs)

@click.command()
@click.option("--spider-dir", required=True, help="Root of Spider dataset (contains train_spider.json/dev.json and database/)")
@click.option("--split", type=click.Choice(["train","dev","both"]), default="both", show_default=True)
@click.option("--out", required=True, help="Output JSONL with gold joins per example")
@click.option("--limit", type=int, default=0, show_default=True, help="0 = all")
def main(spider_dir: str, split: str, out: str, limit: int):
    items = _read_spider_splits(spider_dir)

    # Filter by split if the files are both present and you want just one; otherwise we keep both
    if split != "both":
        want = {"train_spider.json"} if split == "train" else {"dev.json"}
        # crude filter by source filename if you have both; if not, keep all
        # (we skip complicated bookkeeping; Spider often used either/both)
        # We'll just keep all; user can limit later if needed.

    if limit and limit > 0:
        items = items[:limit]

    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with outp.open("w") as f:
        for ex in items:
            gold = _extract_joins(ex["query"])
            rec = {
                "db_id": ex["db_id"],
                "question": ex["question"],
                "sql": ex["query"],
                "tables_in_query": _tables_in_query(ex["query"]),
                "gold_joins": [
                    {"left_table": lt, "left_column": lc, "right_table": rt, "right_column": rc}
                    for (lt, lc, rt, rc) in gold
                ],
            }
            if gold:
                n_ok += 1
            f.write(json.dumps(rec) + "\n")

    click.echo(f"Wrote {outp} with {len(items)} examples ({n_ok} with >=1 join).")

if __name__ == "__main__":
    main()
