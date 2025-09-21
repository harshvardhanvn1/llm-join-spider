from __future__ import annotations
import json, time
from pathlib import Path
import click

from src.joinbench.methods import jaccard_sqlite, containment_sqlite

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

@click.command()
@click.option("--spider-dir", required=True, help="Path to Spider root (contains tables.json and database/)")
@click.option("--pairs", required=True, help="JSONL from build_spider_pairs.py")
@click.option("--method", type=click.Choice(["jaccard", "containment"]), default="jaccard", show_default=True)
@click.option("--threshold", type=float, default=0.05, show_default=True, help="decision threshold on the score")
@click.option("--limit", type=int, default=0, show_default=True, help="optional cap on number of pairs for a quick run")
@click.option("--outdir", type=str, default=None, help="defaults to runs/spider_<method>_<timestamp>")
def main(spider_dir: str, pairs: str, method: str, threshold: float, limit: int, outdir: str | None):
    # load labeled pairs
    items = [json.loads(l) for l in Path(pairs).read_text().splitlines()]
    if limit and limit > 0:
        items = items[:limit]

    # choose predictor
    if method == "jaccard":
        predict = jaccard_sqlite.predict_pair
    else:
        predict = containment_sqlite.predict_pair

    # predict per pair
    y_true, y_pred, preds = [], [], []
    for it in items:
        db_path = _db_file(spider_dir, it["db_id"])
        res = predict(db_path, it["left_table"], it["left_column"], it["right_table"], it["right_column"], threshold=threshold)
        gt = int(it["label"])
        pr = int(res["label"])
        y_true.append(gt)
        y_pred.append(pr)
        rec = {
            **it,
            "gt_label": gt,          # keep gold explicitly
            **res,
            "pred_label": pr         # keep pred explicitly
        }
        # (remove any conflicting 'label' keys if present)
        if "label" in rec and "gt_label" in rec and "pred_label" in rec:
            del rec["label"]
        preds.append(rec)


    # metrics + outputs
    m = _metrics(y_true, y_pred)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(outdir) if outdir else Path("runs") / f"spider_{method}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    click.echo(json.dumps(m, indent=2))
    click.echo(f"wrote → {out}")

if __name__ == "__main__":
    main()
