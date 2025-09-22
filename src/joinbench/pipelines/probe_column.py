from __future__ import annotations
import click
import sqlite3
from pathlib import Path

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

@click.command()
@click.option("--spider-dir", required=True, help="Path to Spider root")
@click.option("--db-id", required=True, help="Database id, e.g., academic")
@click.option("--table",  "table_name", required=True)
@click.option("--column", "column_name", required=True)
@click.option("--n", type=int, default=10, show_default=True, help="How many sample values to show")
def main(spider_dir: str, db_id: str, table_name: str, column_name: str, n: int):
    db_path = Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite not found: {db_path}")

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    try:
        # table exists?
        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?;", (table_name,))
        row = cur.fetchone()
        if not row:
            print(f"[!] table/view not found: {table_name}")
            # show suggestions
            cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view');")
            others = [r[0] for r in cur.fetchall()]
            print("    available:", ", ".join(sorted(others)[:20]), ("... (+more)" if len(others) > 20 else ""))
            return

        # column exists?
        cur.execute(f'PRAGMA table_info({quote_ident(table_name)})')
        cols = [r[1] for r in cur.fetchall()]
        if column_name not in cols:
            print(f"[!] column not found: {table_name}.{column_name}")
            print("    columns:", ", ".join(cols))
            return

        # counts
        cur.execute(f'SELECT COUNT(*) FROM {quote_ident(table_name)}')
        total = cur.fetchone()[0]

        cur.execute(f'SELECT COUNT(*) FROM {quote_ident(table_name)} WHERE {quote_ident(column_name)} IS NOT NULL')
        nn = cur.fetchone()[0]

        cur.execute(f'SELECT COUNT(DISTINCT {quote_ident(column_name)}) FROM {quote_ident(table_name)}')
        distinct = cur.fetchone()[0]

        print(f"{db_id}.{table_name}.{column_name}")
        print(f"  rows(total)={total}, non_null={nn}, distinct={distinct}")

        # sample raw values (no normalization)
        cur.execute(f'SELECT {quote_ident(column_name)} FROM {quote_ident(table_name)} WHERE {quote_ident(column_name)} IS NOT NULL LIMIT {n}')
        sample = [r[0] for r in cur.fetchall()]
        print("  sample:", sample)
    finally:
        con.close()

if __name__ == "__main__":
    main()
