from pathlib import Path
import json, random
from typing import Dict, List

def _idx_to_name_maps(schema: Dict):
    # Use ORIGINAL names to stay close to the DB identifiers; we’ll still resolve later at query time.
    col_original = schema["column_names_original"]  # list of [table_idx, "col name"]
    tables_original = schema["table_names_original"]
    col_map = {}
    for idx, (tid, cname) in enumerate(col_original):
        if tid == -1:  # "*" pseudo-column
            col_map[idx] = ("*", cname)
        else:
            col_map[idx] = (tables_original[tid], cname)
    return col_map

def build_fk_pairs(spider_root: str, out_jsonl: str, negatives_per_pos: int = 1, seed: int = 42) -> str:
    random.seed(seed)
    root = Path(spider_root)
    tables_path = root / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"tables.json not found under: {root}")
    schemas = json.loads(tables_path.read_text())

    pairs: List[Dict] = []
    for schema in schemas:
        db_id = schema["db_id"]
        col_map = _idx_to_name_maps(schema)

        # 1) positives from foreign_keys: [from_col_idx, to_col_idx]
        pos = []
        for a_idx, b_idx in schema.get("foreign_keys", []):
            lt, lc = col_map[a_idx]
            rt, rc = col_map[b_idx]
            if lt == "*" or rt == "*":
                continue
            # keep one direction (dedupe)
            key = (lt, lc, rt, rc)
            rkey = (rt, rc, lt, lc)
            if key in {(p["left_table"], p["left_column"], p["right_table"], p["right_column"]) for p in pos}:
                continue
            if rkey in {(p["left_table"], p["left_column"], p["right_table"], p["right_column"]) for p in pos}:
                continue
            pos.append({
                "db_id": db_id,
                "left_table": lt, "left_column": lc,
                "right_table": rt, "right_column": rc,
                "label": 1
            })

        # 2) negatives: sample non-FK pairs from the same DB
        cols = [(t, c) for (t, c) in [col_map[i] for i in range(len(col_map))] if t != "*"]
        pos_set = set((p["left_table"], p["left_column"], p["right_table"], p["right_column"]) for p in pos)
        pos_set |= set((p["right_table"], p["right_column"], p["left_table"], p["left_column"]) for p in pos)

        needed = negatives_per_pos * max(1, len(pos))
        tries = 0
        neg = []
        while needed > 0 and tries < 10000 and len(cols) >= 2:
            tries += 1
            (lt, lc) = random.choice(cols)
            (rt, rc) = random.choice(cols)
            if lt == rt and lc == rc:
                continue
            if (lt, lc, rt, rc) in pos_set or (rt, rc, lt, lc) in pos_set:
                continue
            neg.append({
                "db_id": db_id,
                "left_table": lt, "left_column": lc,
                "right_table": rt, "right_column": rc,
                "label": 0
            })
            needed -= 1

        pairs.extend(pos)
        pairs.extend(neg)

    outp = Path(out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec) + "\n")
    return str(outp)
