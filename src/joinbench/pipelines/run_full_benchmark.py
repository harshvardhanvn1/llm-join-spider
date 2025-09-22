from __future__ import annotations
import json, time, subprocess, sys, os
from pathlib import Path
import click

def _run(cmd: list[str]):
    click.echo("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.stderr.write(r.stderr)
        raise click.ClickException(f"Command failed: {' '.join(cmd)}")
    if r.stdout:
        print(r.stdout.strip())
    if r.stderr:
        sys.stderr.write(r.stderr)
    return r

@click.command()
@click.option("--spider-dir", required=True)
@click.option("--pairs-out", default="data/benchmarks/spider_pairs.jsonl", show_default=True)
@click.option("--queries-out", default="data/benchmarks/spider_query_joins.jsonl", show_default=True)
@click.option("--jaccard-thr", type=float, default=0.02, show_default=True)
@click.option("--containment-thr", type=float, default=0.02, show_default=True)
@click.option("--llm-thr", type=float, default=0.5, show_default=True)
@click.option("--outdir", default="runs", show_default=True)
def main(spider_dir: str, pairs_out: str, queries_out: str,
         jaccard_thr: float, containment_thr: float, llm_thr: float, outdir: str):
    """
    Full pipeline:
      1) Build schema-pairs JSONL
      2) Build query gold joins JSONL
      3) Run schema-only baselines on pairs (Jaccard, Containment)
      4) Run query baselines (Jaccard, Containment)
      5) Run query LLM benchmark (Gemini, rate-limited)
      6) Summarize to RESULTS.md
    """
    # 1) Build schema pairs (full)
    _run([sys.executable, "-m", "joinbench.pipelines.build_spider_pairs",
          "--spider-dir", spider_dir,
          "--out", pairs_out])

    # 2) Build query gold joins (full)
    _run([sys.executable, "-m", "joinbench.pipelines.build_spider_query_joins",
          "--spider-dir", spider_dir,
          "--out", queries_out])

    # 3) Schema-only baselines on pairs
    _run([sys.executable, "-m", "joinbench.pipelines.run_spider_benchmark",
          "--spider-dir", spider_dir, "--pairs", pairs_out,
          "--method", "jaccard", "--threshold", str(jaccard_thr)])
    _run([sys.executable, "-m", "joinbench.pipelines.run_spider_benchmark",
          "--spider-dir", spider_dir, "--pairs", pairs_out,
          "--method", "containment", "--threshold", str(containment_thr)])

    # 4) Query-conditional baselines (no LLM)
    _run([sys.executable, "-m", "joinbench.pipelines.run_query_baselines",
          "--spider-dir", spider_dir, "--gold", queries_out,
          "--method", "jaccard", "--threshold", str(jaccard_thr)])
    _run([sys.executable, "-m", "joinbench.pipelines.run_query_baselines",
          "--spider-dir", spider_dir, "--gold", queries_out,
          "--method", "containment", "--threshold", str(containment_thr)])

    # 5) Query LLM
    # (Gemini is auto-paced by our shared limiter; set GEMINI_MODEL in .env)
    _run([sys.executable, "-m", "joinbench.pipelines.run_query_join_benchmark",
          "--spider-dir", spider_dir, "--gold", queries_out])

    # 6) Summarize the latest runs (top 1 per method) into RESULTS.md
    _run([sys.executable, "-m", "joinbench.pipelines.summarize_runs",
          "--runs-dir", "runs",
          "--methods", "jaccard,containment,llm",
          "--topk", "1",
          "--out-md", "RESULTS.md"])

    print("\nAll done ✔  See RESULTS.md and the 'runs/' directory for artefacts.")

if __name__ == "__main__":
    main()
