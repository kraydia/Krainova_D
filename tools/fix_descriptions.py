"""
one-off helper: clean up product names and rewrite weak descriptions.

1. removes leftover "TM" words from product names (same rule as strip_symbols)
2. finds weak descriptions: fewer than 6 words, about sales/usage numbers
   instead of the product, the same text used for two products, or listed
   in MIX_UPS below (description belongs to a different product)
3. for each weak one, takes the sentences from the original filing that
   mention the product and asks the LLM to write a new description from them

the filing text comes from the download cache, so no new SEC requests are
needed. a new description is only used if it passes the same length checks.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import Krainova_Assignment_2 as e

# rows where I noticed the description describes a different product
MIX_UPS = {("ABT", "Ensure shake featuring 42 grams of protein")}

# I read every rewrite. For these the LLM's new description was not better
# (muddled with other products or just as vague), so the old one is kept.
KEEP_OLD = {("AAPL", "AirPods 4"), ("AAPL", "AirPods Pro 3"),
            ("DDOG", "Bits Code"), ("DDOG", "Bits Chat"),
            ("CSGP", "Homes.com Ai"), ("LUV", "extra legroom seating")}

# the filing lists several launches in one run-on sentence, so the model
# attached the next product's description (infant formula) to this shake.
# neither LLM pass could fix it, so this description is written by hand
# from the product name only.
MANUAL_DESCRIPTIONS = {
    ("ABT", "Ensure shake featuring 42 grams of protein"):
        "An Ensure nutrition shake with 42 grams of protein.",
}

# descriptions about how well something sold, not what it is
SALES_TALK = re.compile(r"\b(sales|to date|cumulative spend|ended Q\d)\b", re.I)

DESCRIBE_PROMPT = """Here are sentences from an SEC filing by {company}.

{text}

In one sentence under 170 characters, say what "{product}" is or does.
Use only facts from these sentences. Do not mention sales numbers.
Return JSON: {{"product_description": "<sentence>"}}"""

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"product_description": {"type": "string"}},
    "required": ["product_description"],
}


def is_weak(row, desc_counts):
    desc = row["product_description"]
    return (len(desc.split()) < 6
            or SALES_TALK.search(desc)
            or desc_counts[(row["stock_name"], desc)] > 1
            or (row["stock_name"], row["new_product"]) in MIX_UPS)


def filing_text(row, log_rows, ticker_map):
    """get the cached press-release text of the filing this row came from."""
    cik = int(ticker_map[row["stock_name"]][0])
    for log in log_rows:
        if (log["ticker"] == row["stock_name"]
                and log["filing_time"] == row["filing_time"]):
            acc = log["accession"]
            index_url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                         f"{acc.replace('-', '')}/{acc}-index.htm")
            text, _ = e.fetch_filing_text({"index_url": index_url})
            if e.normalise(row["new_product"]).split()[0] in e.normalise(text):
                return text
    return ""


def sentences_about(product, text, limit=2500):
    """
    sentences that mention the product. tries the full name first, then the
    first two words, then the first word, so a common first word like
    "Disney" or "Ensure" doesn't pull in the whole press release.
    """
    words = e.normalise(product).split()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for n in (len(words), 2, 1):
        key = " ".join(words[:n])
        hits = [s for s in sentences if key in e.normalise(s)]
        if hits:
            return " ".join(hits)[:limit]
    return ""


def describe(company, product, text):
    resp = requests.post(e.OLLAMA_URL, json={
        "model": e.DEFAULT_MODEL,
        "messages": [{"role": "user", "content": DESCRIBE_PROMPT.format(
            company=company, product=product, text=text)}],
        "format": DESCRIBE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 120},
    }, timeout=300)
    resp.raise_for_status()
    answer = json.loads(resp.json()["message"]["content"])
    return e.clean_description(answer.get("product_description", ""))


def main():
    out = e.OUTPUT_DIR / "new_products.csv"
    rows = e.read_csv(out)
    log_rows = e.read_csv(e.OUTPUT_DIR / "processing_log.csv")
    ticker_map = e.load_ticker_map()

    # step 1: names
    for row in rows:
        row["new_product"] = e.strip_symbols(row["new_product"], "")

    # step 2 + 3: weak descriptions
    desc_counts = Counter((r["stock_name"], r["product_description"]) for r in rows)
    weak = [r for r in rows if is_weak(r, desc_counts)]
    print(len(weak), "weak descriptions")
    for row in weak:
        key = (row["stock_name"], row["new_product"])
        if key in MANUAL_DESCRIPTIONS:
            row["product_description"] = MANUAL_DESCRIPTIONS[key]
            print(f"  {key[0]} {key[1]}: manual description")
            continue
        if key in KEEP_OLD:
            print(f"  {key[0]} {key[1]}: kept old (rewrite was not better)")
            continue
        text = sentences_about(row["new_product"],
                               filing_text(row, log_rows, ticker_map))
        if not text:
            print("  no filing text for", row["new_product"])
            continue
        new = describe(row["company_name"], row["new_product"], text)
        if len(new.split()) >= 6 and not SALES_TALK.search(new):
            print(f"  {row['stock_name']} {row['new_product']}: "
                  f"{row['product_description']!r} -> {new!r}")
            row["product_description"] = new
        else:
            print(f"  {row['stock_name']} {row['new_product']}: kept "
                  f"(LLM answer too weak: {new!r})")

    out.unlink()
    for row in rows:
        e.append_csv(out, e.CSV_COLUMNS, row)
    # re-apply the manual exclusion list, then rebuild the " | " table
    print("removed", e.apply_manual_exclusions(out), "excluded rows")
    e.write_pipe_table(out, e.OUTPUT_DIR / "new_products.txt")


if __name__ == "__main__":
    main()
