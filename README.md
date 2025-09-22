# Leverage LLM for finding potential SQL Joins

Discovering **SQL join keys** automatically on the [Spider](https://yale-lily.github.io/spider) dataset using:

- **Value-overlap baselines** — Jaccard and Containment on column **values** (not just headers)
- **Query-conditioned LLM** — Gemini selects the equality join edges needed to answer a natural-language question

This repo provides clean CLIs to:

1) Build benchmark inputs (schema column pairs, and gold joins parsed from SQL)  
2) Run baselines and LLMs (schema-only + query-conditioned)  
3) Analyze failures and summarize results

> Works with Spider SQLite DBs  
> Robust to non-UTF8 text values in Spider  
> Paces Gemini calls (supports `gemini-2.0-flash-lite`)

---

## Repo layout

```
src/joinbench/
  data/
    sqlite_utils.py            # robust value loader, identifier helpers
  llm/
    gemini_client.py           # rate-limited Gemini client w/ env-based model
  methods/
    jaccard_sqlite.py          # |A∩B| / |A∪B| on DISTINCT column values
    containment_sqlite.py      # |A∩B| / min(|A|,|B|) on DISTINCT column values
    llm_spider_gemini.py       # schema-only LLM (column-pair classification)
    llm_query_join_gemini.py   # query-conditioned LLM (candidate selection + justification)
  pipelines/
    build_spider_pairs.py          # build schema pairs JSONL (full dataset)
    build_spider_query_joins.py    # parse gold joins from SQL (via sqlglot)
    run_spider_benchmark.py        # eval schema baselines & schema-LLM
    run_query_baselines.py         # eval Jaccard/Containment on query task
    run_query_join_benchmark.py    # eval LLM on query task
    threshold_sweep.py             # threshold sweep per method
    summarize_runs.py              # summarize best runs into RESULTS.md
    run_all_methods.py             # schema baselines wrapper
    run_full_benchmark.py          # full E2E: build inputs + all methods + summary
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dataset layout (Spider)

```
/ABS/PATH/TO/spider/
  database/<db_id>/<db_id>.sqlite
  train_spider.json
  dev.json
```

---

## Configure Gemini

Create a `.env` in the repo root (this file is already gitignored):

```
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL=gemini-2.0-flash-lite
```

You can switch models by changing `GEMINI_MODEL`. The client hot-reloads when env changes.

---

## Quickstart — schema baselines

1) Build schema pairs JSONL (full dataset):

```bash
python -m joinbench.pipelines.build_spider_pairs   --spider-dir "/ABS/PATH/TO/spider"   --out data/benchmarks/spider_pairs.jsonl
```

2) Run Jaccard & Containment:

```bash
python -m joinbench.pipelines.run_spider_benchmark   --spider-dir "/ABS/PATH/TO/spider"   --pairs data/benchmarks/spider_pairs.jsonl   --method jaccard --threshold 0.02

python -m joinbench.pipelines.run_spider_benchmark   --spider-dir "/ABS/PATH/TO/spider"   --pairs data/benchmarks/spider_pairs.jsonl   --method containment --threshold 0.02
```

3) Inspect failures (optional):

```bash
python -m joinbench.pipelines.failure_report   --spider-dir "/ABS/PATH/TO/spider"   --run-dir runs/<run_dir_from_above>   --k 10
```

> Note: “schema-only” means **no natural-language question**. It still uses **column values** from SQLite to compute overlap, not just header strings.

---

## Query-conditioned task

The goal here is: given a **question** and the **tables it uses**, predict which equality joins are needed to answer it.

### 1) Build gold joins per question

We parse Spider’s **gold SQL** using `sqlglot` to extract join edges.

```bash
python -m joinbench.pipelines.build_spider_query_joins   --spider-dir "/ABS/PATH/TO/spider"   --out data/benchmarks/spider_query_joins.jsonl
```

Each JSONL record contains:
- `db_id`, `question`, `sql`
- `tables_in_query`: tables touched by the SQL
- `gold_joins`: list of edges `{left_table,left_column,right_table,right_column}`

### 2) Query-conditioned baselines (no LLM)

We evaluate Jaccard/Containment **only among column pairs** from the tables used in each question.

```bash
python -m joinbench.pipelines.run_query_baselines   --spider-dir "/ABS/PATH/TO/spider"   --gold data/benchmarks/spider_query_joins.jsonl   --method jaccard --threshold 0.02

python -m joinbench.pipelines.run_query_baselines   --spider-dir "/ABS/PATH/TO/spider"   --gold data/benchmarks/spider_query_joins.jsonl   --method containment --threshold 0.02
```

### 3) Query-conditioned LLM (Gemini)

- We build a **small candidate set** of plausible equality joins using schema signals:
  - Declared foreign keys, primary keys, `*_id` patterns, and exact same-name columns
- The model **selects** by **indices** from these candidates (cannot invent)
- We record a concise `justification` and compact `evidence` tags (`pk_fk`, `id_like`, `same_name`, `question_signal`)

```bash
python -m joinbench.pipelines.run_query_join_benchmark   --spider-dir "/ABS/PATH/TO/spider"   --gold data/benchmarks/spider_query_joins.jsonl
```

Outputs:
```
runs/queries_llm_<timestamp>/
  predictions.jsonl    # per-example predictions + explain
  metrics.json         # micro P/R/F1 over edges
  failures.json        # sample mismatches
```

---

## Threshold sweep & summarization

Sweep thresholds (example: Jaccard):

```bash
python -m joinbench.pipelines.threshold_sweep   --spider-dir "/ABS/PATH/TO/spider"   --pairs data/benchmarks/spider_pairs.jsonl   --method jaccard   --thresholds 0.00:0.20:0.02   --limit 200
```

Summarize best runs into `RESULTS.md`:

```bash
python -m joinbench.pipelines.summarize_runs   --runs-dir runs   --methods jaccard,containment,llm   --topk 1   --out-md RESULTS.md
```

---

## Full end-to-end (whole dataset)

This one command rebuilds inputs and runs **all** methods (schema + query) and writes `RESULTS.md`:

```bash
python -m joinbench.pipelines.run_full_benchmark   --spider-dir "/ABS/PATH/TO/spider"   --jaccard-thr 0.02   --containment-thr 0.02   --llm-thr 0.50   --outdir runs
```

The Gemini client respects API quotas with backoff. If you hit daily caps on free tiers, switch to `gemini-2.0-flash-lite` (already the default in this README) or rerun later.

---

## Design choices

- **Baselines are value-based**  
  We compute overlap using **distinct column values** pulled from SQLite (not titles). We also added a **robust text loader** (fetch bytes → decode UTF-8 → Latin-1 → replacement) to handle Spider’s non-UTF8 text.

- **Schema-only vs Query-conditioned**  
  - Schema-only = column-pair classification (no question).  
  - Query-conditioned = restrict to the tables used by the question; for LLM we add a reasoning prompt and candidate selection.

- **LLM reasoning (safe)**  
  We request a short **justification** and **evidence tags**—no chain-of-thought is logged. The model picks from enumerated candidates to minimize hallucinations.

---

## Troubleshooting

- **`sqlite3.OperationalError: Could not decode to UTF-8 …`**  
  Already handled; ensure you’ve reinstalled after pulling: `pip install -e .`.

- **Gemini 429 / quota exceeded**  
  The client retries with backoff and paces requests. Free tiers have daily caps; use `gemini-2.0-flash-lite` or retry later.

- **gRPC warnings (`ALTS creds ignored`)**  
  Harmless noise from Google’s SDK when not on GCP.

---

## Roadmap

- Type gates (block incompatible types)
- FK-ratio features (`|A∩B|/|A|`, `|A∩B|/|B|`)
- Tiny **value samples** per candidate to improve LLM disambiguation
- Optional strict name-similarity feature flag
- Per-DB breakdowns and plots

---

## References

- Yu et al., **Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task** (EMNLP 2018).  
- *LakeBench* (attached in project files): benchmarking LLMs on compositional DB tasks.  
- **2306.09610v3** (attached in project files): techniques for schema linking / join detection with LLMs.  
- **CorpusCrew Final Report** (attached in project files): internal report of early findings and failures.

> These were used for methodological guidance and benchmarking context. Local copies are included alongside this repo’s `data/` folder per your notes.

---


