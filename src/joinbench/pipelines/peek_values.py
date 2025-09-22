from __future__ import annotations
import click
from pathlib import Path
from joinbench.data.sqlite_utils import (
    resolve_table_name, resolve_column_name, load_column_values_sqlite
)

@click.command()
@click.option("--spider-dir", required=True, help="Path to Spider root")
@click.option("--db-id", required=True, help="Database id, e.g., academic")
@click.option("--left-table",  required=True)
@click.option("--left-column", required=True)
@click.option("--right-table",  required=True)
@click.option("--right-column", required=True)
@click.option("--n", type=int, default=10, show_default=True, help="How many distinct values to show per side")
def main(spider_dir: str, db_id: str,
         left_table: str, left_column: str,
         right_table: str, right_column: str, n: int):
    db_path = Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite not found: {db_path}")

    # resolve identifiers
    lt_res = resolve_table_name(str(db_path), left_table)
    lc_res = resolve_column_name(str(db_path), lt_res, left_column)
    rt_res = resolve_table_name(str(db_path), right_table)
    rc_res = resolve_column_name(str(db_path), rt_res, right_column)

    # load a small sample
    L = sorted(list(load_column_values_sqlite(str(db_path), lt_res, lc_res)))[:n]
    R = sorted(list(load_column_values_sqlite(str(db_path), rt_res, rc_res)))[:n]

    print("\nResolved identifiers:")
    print(f"  left : {left_table}.{left_column}  ->  {lt_res}.{lc_res}  (|L|={len(L)})")
    print(f"  right: {right_table}.{right_column} ->  {rt_res}.{rc_res} (|R|={len(R)})")

    print("\nLeft sample values:", L)
    print("Right sample values:", R)

if __name__ == "__main__":
    main()
