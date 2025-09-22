from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, List, Tuple, Set
import click

from joinbench.methods.llm_query_join_gemini import predict_joins

def _db_file(spider_dir: str, db_id: str) -> str:
    p = Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(f"SQLite not found: {p}")
    return str(p)

def _norm_edge(e: Dict) -> Tuple[str,str,str,str]:
    lt, lc = e["left_table"], e["left_column"]
    rt, rc = e["right_table"], e["right_column"]
    if (rt, rc) < (lt, lc):
        lt, lc, rt, rc = rt, rc, lt, lc
    return (lt, lc, rt, rc)

def _edge_set(lst: List[Dict]) -> Set[Tuple[str,str,str,str]]:
    return { _norm_edge(e) for e in lst }

def _metrics(true_edges: List[Set[Tuple]], pred_edges: List[Set[Tuple]]):
    # micro over edges
    tp = fp = fn = tn = 0
    for T, P in zip(true_edges, pred_edges):
        tp += len(T & P)
        fp += len(P - T)
        fn += len(T - P)
        # tn is not well-defined without closed world of non-edges; omit from micro.
    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall    = tp / (tp+fn) if (tp+fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edge_counts": {"tp": tp, "fp": fp, "fn": fn}
    }

@click.command()
@click.option("--spider-dir", required=True, help="Root of Spider (has database/ and train/dev jsons)")
@click.option("--gold", required=True, help="JSONL from build_spider_query_joins.py")
@click.option("--limit", type=int, default=0, show_default=True, help="0 = all examples")
@click.option("--outdir", default="runs", show_default=True)
def main(spider_dir: str, gold: str, limit: int, outdir: str):
    items = [json.loads(l) for l in Path(gold).read_text().splitlines()]
    if limit and limit > 0:
        items = items[:limit]

    true_sets: List[Set[Tuple]] = []
    pred_sets: List[Set[Tuple]] = []
    preds_out: List[Dict] = []

    for ex in items:
        db_id = ex["db_id"]
        db_path = _db_file(spider_dir, db_id)
        tables = ex.get("tables_in_query", [])
        question = ex.get("question","")
        gold_edges = _edge_set(ex.get("gold_joins", []))

        res = predict_joins(db_path, db_id, question, tables)
        pred_edges = _edge_set(res.get("pred_joins", []))

        true_sets.append(gold_edges)
        pred_sets.append(pred_edges)

        preds_out.append({
            "db_id": db_id,
            "question": question,
            "tables_in_query": tables,
            "gold_joins": sorted(list(gold_edges)),
            "pred_joins": sorted(list(pred_edges)),
            "explain": res.get("explain",""),
        })

    m = _metrics(true_sets, pred_sets)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(outdir) / f"queries_llm_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    # Save predictions & metrics
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds_out))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))

    # Quick failure report (top 10 with most misses)
    misses = []
    for p in preds_out:
        T = set(map(tuple, p["gold_joins"]))
        P = set(map(tuple, p["pred_joins"]))
        if T != P:
            misses.append({
                "db_id": p["db_id"],
                "tables_in_query": p["tables_in_query"],
                "question": p["question"],
                "gold": sorted(T),
                "pred": sorted(P),
                "missed": sorted(T - P),
                "extra": sorted(P - T),
                "explain": p.get("explain",""),
            })
    misses = misses[:10]
    (out / "failures.json").write_text(json.dumps(misses, indent=2))

    click.echo(json.dumps(m))
    click.echo(f"wrote → {out}")

if __name__ == "__main__":
    main()
