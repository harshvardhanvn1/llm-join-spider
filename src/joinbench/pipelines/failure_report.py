from __future__ import annotations
import json
from pathlib import Path
import click
from tabulate import tabulate
from joinbench.data.sqlite_utils import load_column_values_resolved, resolve_table_name, resolve_column_name

@click.command()
@click.option("--spider-dir", required=True, help="Path to Spider root")
@click.option("--run-dir", required=True, help="Path to a run folder (contains predictions.jsonl)")
@click.option("--k", type=int, default=5, show_default=True, help="How many examples to show per split (FN/FP)")
def main(spider_dir: str, run_dir: str, k: int):
    rundir = Path(run_dir)
    preds_path = rundir / "predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {rundir}")
    items = [json.loads(l) for l in preds_path.read_text().splitlines()]

    # Split by outcome using explicit fields added by the runner
    FNs = [p for p in items if int(p["gt_label"]) == 1 and int(p["pred_label"]) == 0]
    FPs = [p for p in items if int(p["gt_label"]) == 0 and int(p["pred_label"]) == 1]

    def sample_row(p):
        db = Path(spider_dir) / "database" / p["db_id"] / f'{p["db_id"]}.sqlite'
        lt, lc = p["left_table"], p["left_column"]
        rt, rc = p["right_table"], p["right_column"]
        # take small samples to display
        L = sorted(list(load_column_values_resolved(str(db), lt, lc)))[:5]
        R = sorted(list(load_column_values_resolved(str(db), rt, rc)))[:5]
        inter = sorted(list(set(L) & set(R)))[:5]
        return [p["db_id"], f'{lt}.{lc}', f'{rt}.{rc}', p.get("score", 0.0), p.get("explain", ""), ", ".join(inter), ", ".join(L[:3]), ", ".join(R[:3])]

    def show(title, arr):
        print(f"\n# {title} (showing up to {k})")
        print(tabulate([sample_row(p) for p in arr[:k]],
                       headers=["db", "left", "right", "score", "explain", "overlap(∩) sample", "left sample", "right sample"],
                       tablefmt="github"))

    show("False Negatives (gt=1, pred=0)", FNs)
    show("False Positives (gt=0, pred=1)", FPs)

if __name__ == "__main__":
    main()
