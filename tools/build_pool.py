"""
one-off helper used once while building the final dataset (see "Problems I
ran into" in the report).

an earlier version of the second LLM pass rejected almost every candidate.
Its rejects were logged, so this script merges every first-stage candidate
(kept + rejected) back into output/new_products.csv. The fixed category
verifier is then re-applied with:
    python Krainova_Assignment_2.py --verify-existing --target-rows 0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import Krainova_Assignment_2 as e

out = e.OUTPUT_DIR / "new_products.csv"
names = {k: e.pretty_company(v[1]) for k, v in e.load_ticker_map().items()}
pool, seen = [], set()
for row in e.read_csv(e.OUTPUT_DIR / "rejected_candidates.csv") + e.read_csv(out):
    key = (row["stock_name"], e.normalise(row["new_product"]))
    if key in seen:
        continue
    seen.add(key)
    pool.append({"company_name": row.get("company_name") or names.get(row["stock_name"], row["stock_name"]),
                 "stock_name": row["stock_name"], "filing_time": row["filing_time"],
                 "new_product": row["new_product"],
                 "product_description": row["product_description"]})
pool.sort(key=lambda r: (r["stock_name"], r["filing_time"]))
out.unlink()
for row in pool:
    e.append_csv(out, e.CSV_COLUMNS, row)
(e.OUTPUT_DIR / "rejected_candidates.csv").unlink()
print(len(pool), "candidates in pool")
