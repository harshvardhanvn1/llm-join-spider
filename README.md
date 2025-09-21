# LLM-Join-Spider (Spider-only baseline)

Find potential SQL joins on Spider. Step 1 builds FK-based gold pairs (positives) and adds same-DB negatives.
Step 2 runs baselines (Jaccard/Containment) directly on Spider SQLite DBs.

## Quickstart
```bash
pip install -e .
python -m joinbench.pipelines.build_spider_pairs \
  --spider-dir /ABS/PATH/TO/spider \
  --out data/benchmarks/spider_pairs.jsonl \
  --negatives-per-pos 1
