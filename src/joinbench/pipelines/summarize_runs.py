from __future__ import annotations
import json
from pathlib import Path
import click
from tabulate import tabulate

def _collect_runs(runs_dir: Path, method: str):
    rows = []
    # accept both sweep_* and spider_* (single-run) directories
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir(): continue
        name = p.name
        if method not in name: continue
        metrics_json = p / "metrics.json"
        best_json    = p / "best.json"
        sweep_json   = p / "sweep.json"
        rec = {"run_dir": str(p)}
        try:
            if metrics_json.exists():
                m = json.loads(metrics_json.read_text())
                rec.update({"kind":"single","f1":m["f1"],"precision":m["precision"],"recall":m["recall"],"accuracy":m["accuracy"]})
            elif best_json.exists():
                b = json.loads(best_json.read_text())
                rec.update({"kind":"sweep-best","f1":b["f1"],"precision":b["precision"],"recall":b["recall"],"accuracy":b["accuracy"],"threshold":b["threshold"]})
            elif sweep_json.exists():
                S = json.loads(sweep_json.read_text())
                if S:
                    b = max(S, key=lambda r: r["f1"])
                    rec.update({"kind":"sweep-list","f1":b["f1"],"precision":b["precision"],"recall":b["recall"],"accuracy":b["accuracy"],"threshold":b["threshold"]})
                else:
                    rec.update({"kind":"sweep-list","f1":0,"precision":0,"recall":0,"accuracy":0})
            else:
                continue
            rows.append(rec)
        except Exception:
            continue
    return rows

@click.command()
@click.option("--runs-dir", default="runs", show_default=True)
@click.option("--methods", default="jaccard,containment,llm", show_default=True,
              help="comma list from {jaccard,containment,llm}")
@click.option("--topk", type=int, default=1, show_default=True)
@click.option("--out-md", type=str, default=None, help="If set, write a Markdown summary file")
def main(runs_dir: str, methods: str, topk: int, out_md: str | None):
    runs_dir = Path(runs_dir)
    methods = [m.strip() for m in methods.split(",") if m.strip()]
    all_tables = []
    md_parts = []

    for m in methods:
        rows = _collect_runs(runs_dir, m)
        if not rows:
            print(f"(no runs found for {m})")
            continue
        rows_sorted = sorted(rows, key=lambda r: r["f1"], reverse=True)
        best = rows_sorted[:topk]
        tbl = [[r["kind"], r.get("threshold","-"), f'{r["f1"]:.3f}', f'{r["precision"]:.3f}', f'{r["recall"]:.3f}', f'{r["accuracy"]:.3f}', r["run_dir"]] for r in best]
        print(f"\n# {m.upper()} — top {len(best)}")
        print(tabulate(tbl, headers=["kind","thr","F1","P","R","Acc","run"], tablefmt="github"))
        all_tables.append((m, tbl))
        md_parts.append(f"\n## {m.upper()} — top {len(best)}\n" + tabulate(tbl, headers=["kind","thr","F1","P","R","Acc","run"], tablefmt="github"))

    if out_md:
        Path(out_md).write_text("# Results summary\n" + "\n".join(md_parts))
        print(f"\nWrote Markdown summary to {out_md}")

if __name__ == "__main__":
    main()
