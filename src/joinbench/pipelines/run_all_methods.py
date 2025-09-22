from __future__ import annotations
import json, time
from pathlib import Path
import click

from joinbench.methods import jaccard_sqlite, containment_sqlite
from joinbench.methods import llm_spider_gemini  # uses shared rate limiter

def _db_file(spider_dir: str, db_id: str) -> str:
    p = Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(f"SQLite not found: {p}")
    return str(p)

def _metrics(y_true, y_pred):
    tp = sum(1 for t,p in zip(y_true, y_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(y_true, y_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(y_true, y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true, y_pred) if t==1 and p==0)
    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall    = tp / (tp+fn) if (tp+fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    accuracy  = (tp+tn) / max(len(y_true), 1)
    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "confusion_matrix": [[tn, fp],[fn, tp]],
        "counts": {"tp":tp,"fp":fp,"fn":fn,"tn":tn}
    }

def _run_one(method: str, threshold: float, items, spider_dir: str, out_root: Path):
    if method == "jaccard":     predict = jaccard_sqlite.predict_pair
    elif method == "containment": predict = containment_sqlite.predict_pair
    elif method == "llm":
        predict = lambda db, lt, lc, rt, rc, threshold: llm_spider_gemini.predict_pair(
            db, lt, lc, rt, rc, threshold=threshold
        )
    else:
        raise click.ClickException(f"Unknown method: {method}")

    y_true, y_pred, preds = [], [], []
    for it in items:
        db_path = _db_file(spider_dir, it["db_id"])
        res = predict(db_path, it["left_table"], it["left_column"], it["right_table"], it["right_column"], threshold=threshold)
        gt = int(it["label"]); pr = int(res["label"])
        y_true.append(gt); y_pred.append(pr)
        rec = {**it, "gt_label": gt, **res, "pred_label": pr}
        if "label" in rec: del rec["label"]
        preds.append(rec)

    m = _metrics(y_true, y_pred)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = out_root / f"spider_{method}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "meta.json").write_text(json.dumps({"method": method, "threshold": threshold, "n": len(items)}, indent=2))
    click.echo(f"{method}: {json.dumps(m)}")
    click.echo(f"wrote → {out}")
    return out

@click.command()
@click.option("--spider-dir", required=True)
@click.option("--pairs", required=True)
@click.option("--methods", default="jaccard,containment,llm", show_default=True)
@click.option("--jaccard-thr", type=float, default=0.05, show_default=True)
@click.option("--containment-thr", type=float, default=0.05, show_default=True)
@click.option("--llm-thr", type=float, default=0.5, show_default=True)
@click.option("--limit", type=int, default=0, show_default=True, help="0 = all pairs")
@click.option("--outdir", type=str, default="runs", show_default=True)
def main(spider_dir: str, pairs: str, methods: str,
         jaccard_thr: float, containment_thr: float, llm_thr: float,
         limit: int, outdir: str):
    items = [json.loads(l) for l in Path(pairs).read_text().splitlines()]
    if limit and limit > 0: items = items[:limit]
    out_root = Path(outdir)

    methods = [m.strip() for m in methods.split(",") if m.strip()]
    for m in methods:
        thr = jaccard_thr if m=="jaccard" else containment_thr if m=="containment" else llm_thr
        _run_one(m, thr, items, spider_dir, out_root)

if __name__ == "__main__":
    main()
