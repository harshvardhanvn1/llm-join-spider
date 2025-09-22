from __future__ import annotations
import json, time, itertools
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sqlite3
import click

from joinbench.data.sqlite_utils import quote_ident
from joinbench.methods import jaccard_sqlite, containment_sqlite

def _db_file(spider_dir: str, db_id: str) -> str:
    p = Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(f"SQLite not found: {p}")
    return str(p)

def _table_columns(db_path: str, table: str) -> List[str]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(f'PRAGMA table_info({quote_ident(table)})')
        return [r[1] for r in cur.fetchall()]
    finally:
        con.close()

def _norm_edge(lt: str, lc: str, rt: str, rc: str) -> Tuple[str,str,str,str]:
    if (rt, rc) < (lt, lc):
        lt, lc, rt, rc = rt, rc, lt, lc
    return (lt, lc, rt, rc)

def _edge_set(lst: List[Dict]) -> Set[Tuple[str,str,str,str]]:
    out = set()
    for e in lst:
        out.add(_norm_edge(e["left_table"], e["left_column"], e["right_table"], e["right_column"]))
    return out

def _metrics(true_edges: List[Set[Tuple]], pred_edges: List[Set[Tuple]]):
    tp = fp = fn = 0
    for T, P in zip(true_edges, pred_edges):
        tp += len(T & P)
        fp += len(P - T)
        fn += len(T - P)
    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall    = tp / (tp+fn) if (tp+fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "edge_counts": {"tp": tp, "fp": fp, "fn": fn}}

@click.command()
@click.option("--spider-dir", required=True)
@click.option("--gold", required=True, help="JSONL produced by build_spider_query_joins.py")
@click.option("--method", type=click.Choice(["jaccard","containment"]), required=True)
@click.option("--threshold", type=float, required=True)
@click.option("--limit", type=int, default=0, show_default=True)
@click.option("--outdir", default="runs", show_default=True)
def main(spider_dir: str, gold: str, method: str, threshold: float, limit: int, outdir: str):
    items = [json.loads(l) for l in Path(gold).read_text().splitlines()]
    if limit and limit > 0:
        items = items[:limit]

    predict = jaccard_sqlite.predict_pair if method == "jaccard" \
              else containment_sqlite.predict_pair

    preds_out: List[Dict] = []
    true_sets: List[Set[Tuple]] = []
    pred_sets: List[Set[Tuple]] = []

    for ex in items:
        db_id = ex["db_id"]
        db_path = _db_file(spider_dir, db_id)
        tables = ex.get("tables_in_query", [])
        gold_edges = _edge_set(ex.get("gold_joins", []))

        # build candidate column pairs only across different tables in this question
        table_cols = {t: _table_columns(db_path, t) for t in tables}
        candidate_edges = []
        for (t1, t2) in itertools.combinations(tables, 2):
            for c1 in table_cols[t1]:
                for c2 in table_cols[t2]:
                    res = predict(db_path, t1, c1, t2, c2, threshold=threshold)
                    if int(res["label"]) == 1:
                        candidate_edges.append(_norm_edge(t1, c1, t2, c2))

        pred_edges = set(candidate_edges)

        preds_out.append({
            "db_id": db_id,
            "question": ex.get("question",""),
            "tables_in_query": tables,
            "gold_joins": sorted(list(gold_edges)),
            "pred_joins": sorted(list(pred_edges)),
            "method": method,
            "threshold": threshold
        })
        true_sets.append(gold_edges)
        pred_sets.append(pred_edges)

    m = _metrics(true_sets, pred_sets)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(outdir) / f"queries_{method}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds_out))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "meta.json").write_text(json.dumps({"method": method, "threshold": threshold, "n": len(items)}, indent=2))
    click.echo(json.dumps(m))
    click.echo(f"wrote → {out}")

if __name__ == "__main__":
    main()
