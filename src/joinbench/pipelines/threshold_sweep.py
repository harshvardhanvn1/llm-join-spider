from __future__ import annotations
import json, time
from pathlib import Path
from typing import List
import click

from joinbench.methods import jaccard_sqlite, containment_sqlite
from joinbench.methods import llm_spider_gemini  # requires our Gemini client already in place

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

def _choose_predict(method: str):
    if method == "jaccard":     return jaccard_sqlite.predict_pair
    if method == "containment": return containment_sqlite.predict_pair
    if method == "llm":
        return lambda db, lt, lc, rt, rc, threshold: llm_spider_gemini.predict_pair(
            db, lt, lc, rt, rc, threshold=threshold
        )
    raise click.ClickException(f"Unknown method: {method}")

def _iter_thresholds(thr: str) -> List[float]:
    """
    thr can be:
      - a comma list: "0.01,0.05,0.1"
      - a range: "start:stop:step" e.g. "0.00:0.50:0.05"
    """
    thr = thr.strip()
    if ":" in thr:
        s, e, st = [float(x) for x in thr.split(":")]
        vals = []
        v = s
        # be robust to FP accumulation
        while v <= e + 1e-12:
            vals.append(round(v, 6))
            v += st
        return vals
    return [float(x) for x in thr.split(",") if x.strip()]

@click.command()
@click.option("--spider-dir", required=True)
@click.option("--pairs", required=True, help="JSONL from build_spider_pairs.py")
@click.option("--method", type=click.Choice(["jaccard","containment","llm"]), required=True)
@click.option("--thresholds", default="0.00:0.50:0.05", show_default=True,
              help="Comma list or range 'start:stop:step'")
@click.option("--limit", type=int, default=0, show_default=True)
@click.option("--outdir", type=str, default=None)
def main(spider_dir: str, pairs: str, method: str, thresholds: str, limit: int, outdir: str | None):
    items = [json.loads(l) for l in Path(pairs).read_text().splitlines()]
    if limit and limit > 0: items = items[:limit]

    predict = _choose_predict(method)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(outdir) if outdir else Path("runs") / f"sweep_{method}_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for thr in _iter_thresholds(thresholds):
        y_true, y_pred = [], []
        for it in items:
            db_path = _db_file(spider_dir, it["db_id"])
            res = predict(db_path, it["left_table"], it["left_column"],
                          it["right_table"], it["right_column"], threshold=thr)
            y_true.append(int(it["label"]))
            y_pred.append(int(res["label"]))
        m = _metrics(y_true, y_pred)
        m_row = {"threshold": thr, **m}
        results.append(m_row)
        click.echo(f"{method} thr={thr:.3f} => F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")

    # persist
    (out / "sweep.json").write_text(json.dumps(results, indent=2))
    # small CSV for convenience
    csv = "threshold,precision,recall,f1,accuracy,tp,fp,fn,tn\n"
    for r in results:
        c = r["counts"]
        csv += f'{r["threshold"]},{r["precision"]:.6f},{r["recall"]:.6f},{r["f1"]:.6f},{r["accuracy"]:.6f},{c["tp"]},{c["fp"]},{c["fn"]},{c["tn"]}\n'
    (out / "sweep.csv").write_text(csv)

    # best-by-F1
    best = max(results, key=lambda r: r["f1"])
    (out / "best.json").write_text(json.dumps(best, indent=2))
    click.echo(f"BEST {method}: thr={best['threshold']} F1={best['f1']:.3f} (saved to {out})")

if __name__ == "__main__":
    main()
