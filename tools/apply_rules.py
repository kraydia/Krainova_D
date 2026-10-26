"""
one-off helper: apply the name rules and company-level de-duplication that
were added to Krainova_Assignment_2.py last to the rows already saved, then
rewrite output/new_products.txt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import Krainova_Assignment_2 as e

out = e.OUTPUT_DIR / "new_products.csv"
rows, seen, dropped = [], set(), []
for row in e.read_csv(out):
    key = (row["company_name"], e.normalise(row["new_product"]))
    if e.NOT_PRODUCT_WORDS.search(e.normalise(row["new_product"])) or key in seen:
        dropped.append(f'{row["stock_name"]}: {row["new_product"]}')
        continue
    seen.add(key)
    rows.append(row)
out.unlink()
for row in rows:
    e.append_csv(out, e.CSV_COLUMNS, row)
e.write_pipe_table(out, e.OUTPUT_DIR / "new_products.txt")
print("dropped:", dropped)
print(len(rows), "rows,", len({r["stock_name"] for r in rows}), "tickers")
