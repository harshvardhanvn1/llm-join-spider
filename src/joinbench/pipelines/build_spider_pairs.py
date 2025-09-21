import click
from pathlib import Path
from joinbench.data.spider_fk_pairs import build_fk_pairs

@click.command()
@click.option("--spider-dir", required=True, type=str, help="Path to Spider root (contains tables.json, database/)")
@click.option("--out", default="data/benchmarks/spider_pairs.jsonl", show_default=True)
@click.option("--negatives-per-pos", default=1, show_default=True, type=int)
def main(spider_dir: str, out: str, negatives_per_pos: int):
    path = build_fk_pairs(spider_dir, out, negatives_per_pos=negatives_per_pos)
    click.echo(f"pairs → {path}")

if __name__ == "__main__":
    main()
