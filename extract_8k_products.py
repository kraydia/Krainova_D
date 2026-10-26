"""
extract_8k_products.py
======================

extract new product announcements from SEC Form 8-K filings with a local LLM.

pipeline (one pass per S&P 500 company):
    1. Map ticker -> CIK with https://www.sec.gov/files/company_tickers.json
    2. List recent 8-K filings with the EDGAR Atom feed
       (browse-edgar?action=getcompany&CIK=...&type=8-K&output=atom)
    3. Download the filing's press-release exhibits (EX-99.x) or main 8-K
       document and convert the HTML to plain text with BeautifulSoup
    4. Keep only passages that talk about launches / introductions
       (keyword filter) so the prompt stays small
    5. Run Named Entity Recognition (spaCy, ORG / PRODUCT) on those passages
    6. Ask a local Ollama model (default: llama3.2) to return JSON with the
       new products, using the NER entities as hints
    7. Validate the answer (product name and quoted evidence must occur in
       the filing, description < 180 characters)
    8. Second LLM pass: classify each candidate as PRODUCT / DEAL / PROGRAM /
       FACILITY / OTHER and keep only PRODUCT rows in the output CSV

outputs:
    output/new_products.csv     company_name, stock_name, filing_time,
                                new_product, product_description
    output/new_products.txt     the same table, values separated by " | "
    output/processing_log.csv   every filing looked at, with status/timing
    output/rejected_candidates.csv  candidates dropped by the second pass

usage:
    set SEC_USER_AGENT=Your Name your.email@example.com
    ollama pull llama3.2
    python extract_8k_products.py --target-rows 100

the run can be stopped and restarted: filings already in the log are skipped.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# the EDGAR Atom feed is XML; html.parser reads it fine without needing lxml.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# the SEC asks every automated client to identify itself with a contact
# address (https://www.sec.gov/os/accessing-edgar-data). Set SEC_USER_AGENT.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "8K-Product-Extractor student-project contact@example.com"
)
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SEC_MIN_INTERVAL = 0.15          # seconds between requests (SEC limit is 10/s)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FEED_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            "&CIK={cik}&type=8-K&count={count}&output=atom")
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# 8-K items that usually carry press releases. Product launches are almost
# always furnished under Item 7.01 (Reg FD) or 8.01 (Other Events); earnings
# releases (Item 2.02) often mention products launched during the quarter.
RELEVANT_ITEMS = ("7.01", "8.01", "2.02")

# words that signal a product announcement; used to pick passages for the LLM.
LAUNCH_KEYWORDS = re.compile(
    r"\b(launch\w*|introduc\w*|unveil\w*|debut\w*|announc\w*|"
    r"new (?:product|platform|service|solution|model|device|drug|therapy|version)\w*|"
    r"now available|general availability|generally available|roll(?:ed|s|ing)? out|"
    r"approv\w*|released?)\b",
    re.IGNORECASE,
)

MAX_PROMPT_CHARS = 6000          # text budget per filing sent to the LLM
MAX_DESCRIPTION_CHARS = 180      # assignment limit for product_description

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
CSV_COLUMNS = ["company_name", "stock_name", "filing_time",
               "new_product", "product_description"]
LOG_COLUMNS = ["ticker", "company_name", "accession", "filing_time", "items",
               "doc_url", "status", "n_products", "ner_orgs", "ner_products",
               "llm_seconds"]
REJECTED_COLUMNS = ["stock_name", "filing_time", "new_product",
                    "product_description", "reason"]


# ---------------------------------------------------------------------------
# HTTP helpers (rate-limited and cached on disk so reruns are cheap)
# ---------------------------------------------------------------------------

_last_request = 0.0


def http_get(url, headers=None, use_cache=True):
    """return the body of `url` as text, using a disk cache and a rate limit."""
    global _last_request
    CACHE_DIR.mkdir(exist_ok=True)
    # each URL is saved under a file named after its hash, so a second run
    # reads documents from disk instead of downloading them again.
    cache_file = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".cache")
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")

    # wait if the previous request was too recent (SEC allows 10 per second).
    wait = SEC_MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        _last_request = time.time()
        resp = requests.get(url, headers=headers or SEC_HEADERS, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt * 2)        # back off and retry
            continue
        resp.raise_for_status()
        text = resp.text
        if use_cache:
            cache_file.write_text(text, encoding="utf-8")
        return text
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# step 1: company universe (S&P 500) and ticker -> CIK mapping
# ---------------------------------------------------------------------------

def load_sp500_tickers():
    """scrape the S&P 500 constituent tickers from Wikipedia."""
    html = http_get(SP500_URL, headers={"User-Agent": SEC_USER_AGENT})
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if cells:
            # Wikipedia writes BRK.B, SEC writes BRK-B
            tickers.append(cells[0].get_text(strip=True).replace(".", "-"))
    return tickers


def load_ticker_map():
    """return {ticker: (cik_10_digits, company_title)} from the SEC file."""
    data = json.loads(http_get(TICKERS_URL))
    return {
        row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"])
        for row in data.values()
    }


# ---------------------------------------------------------------------------
# step 2-3: list 8-K filings and download their text
# ---------------------------------------------------------------------------

def list_8k_filings(cik, count):
    """parse the EDGAR Atom feed into a list of filing dicts (newest first)."""
    # not cached: the feed changes whenever the company files something new.
    xml = http_get(FEED_URL.format(cik=cik, count=count), use_cache=False)
    soup = BeautifulSoup(xml, "html.parser")   # tolerant of EDGAR's XML
    filings = []
    # every <entry> in the feed is one filing; copy out the fields we need.
    for entry in soup.find_all("entry"):
        def field(tag):
            node = entry.find(tag)
            return node.get_text(strip=True) if node else ""
        filings.append({
            "accession": field("accession-number"),
            "form": field("filing-type"),
            "filing_date": field("filing-date"),
            "timestamp": field("updated"),
            "items": field("items-desc"),
            "index_url": field("filing-href"),
        })
    return filings


def get_filing_documents(index_url):
    """return [(doc_type, url)] for the documents listed on a filing index page."""
    soup = BeautifulSoup(http_get(index_url), "html.parser")
    table = soup.find("table", {"class": "tableFile"})
    docs = []
    if table is None:
        return docs
    # table columns: Seq | Description | Document (link) | Type | Size
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 4 or not cells[2].find("a"):
            continue
        href = cells[2].find("a")["href"]
        if href.startswith("/ix?doc="):            # inline-XBRL viewer link
            href = href[len("/ix?doc="):]
        if not href.lower().endswith((".htm", ".html", ".txt")):
            continue
        docs.append((cells[3].get_text(strip=True).upper(),
                     "https://www.sec.gov" + href))
    return docs


def html_to_text(html):
    """strip tags, scripts and XBRL headers; collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "ix:header", "head"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def fetch_filing_text(filing):
    """download the press-release exhibits (EX-99.x); fall back to the 8-K body."""
    docs = get_filing_documents(filing["index_url"])
    # press releases are attached as Exhibit 99.1, 99.2, ...; the 8-K form
    # itself is mostly legal boilerplate, so it is only used as a fallback.
    exhibits = [url for typ, url in docs if typ.startswith("EX-99")]
    chosen = exhibits or [url for typ, url in docs if typ.startswith("8-K")][:1]
    text = " ".join(html_to_text(http_get(url)) for url in chosen)
    return text, (chosen[0] if chosen else filing["index_url"])


# ---------------------------------------------------------------------------
# step 4: keep only passages that look like product announcements
# ---------------------------------------------------------------------------

def select_launch_passages(text, budget=MAX_PROMPT_CHARS):
    """
    split the text into sentences and keep each keyword sentence plus its
    neighbours. The headline (first ~600 chars) is always kept because press
    releases put the announcement there.
    """
    # rough sentence split: break after . ! or ? followed by a space.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keep = set()
    for i, sentence in enumerate(sentences):
        if LAUNCH_KEYWORDS.search(sentence):
            # the sentence before / after often names or describes the product.
            keep.update({i - 1, i, i + 1})
    if not keep:
        return ""          # no launch words at all -> skip the LLM call
    selected = [sentences[i] for i in sorted(keep) if 0 <= i < len(sentences)]
    passage = text[:600] + " ... " + " ".join(selected)
    return passage[:budget]


# ---------------------------------------------------------------------------
# step 5: Named Entity Recognition
# ---------------------------------------------------------------------------

_nlp = None


def load_ner_model():
    """load spaCy's English pipeline (python -m spacy download en_core_web_sm)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except (ImportError, OSError):
            print("WARNING: spaCy/en_core_web_sm not installed; "
                  "NER hints disabled (the LLM still performs entity tagging).")
            _nlp = False
    return _nlp


def run_ner(text):
    """return (orgs, products): most frequent ORG and PRODUCT entities."""
    nlp = load_ner_model()
    if not nlp:
        return [], []
    # count how often each ORG / PRODUCT name appears; the most frequent ones
    # are passed to the LLM as hints.
    counts = {"ORG": {}, "PRODUCT": {}}
    for ent in nlp(text).ents:
        if ent.label_ in counts:
            name = ent.text.strip()
            counts[ent.label_][name] = counts[ent.label_].get(name, 0) + 1

    def top(label, n):
        ranked = sorted(counts[label].items(), key=lambda kv: -kv[1])
        return [name for name, _ in ranked[:n]]
    return top("ORG", 5), top("PRODUCT", 10)


# ---------------------------------------------------------------------------
# step 6: query the local LLM through Ollama
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst performing named entity recognition
on SEC Form 8-K filings. Entity types: ORGANIZATION (the company) and PRODUCT
(a newly launched or announced product, service, drug, device, software or platform).

Return ONLY JSON in this exact shape:
{"company_name": "<ORGANIZATION>",
 "announces_new_product": true or false,
 "products": [{"new_product": "<PRODUCT name as written in the text>",
               "product_description": "<one sentence, under 180 characters>",
               "evidence": "<the sentence from the text that announces it>"}]}

Rules:
- A PRODUCT is a specific, named offering, e.g. "MacBook Pro", "Zepbound",
  "Blackwell Ultra", "Copilot Studio". Generic phrases such as "AI models",
  "new solutions" or "our platform" are NOT products.
- Include a product only if THIS filing announces that it is new: launched,
  introduced, unveiled, released, approved, or made available.
- Do NOT list financial results, dividends, debt or share offerings, mergers,
  executive appointments, legal matters, or products that already existed.
- Business segments, reporting units and revenue lines in investor
  presentations (e.g. "Intelligent Cloud", "LinkedIn Premium") are NOT new
  products.
- Also NOT products: securities, acquisitions or divestitures, partnerships,
  facilities, webcasts or websites, and products merely mentioned because
  their sales grew.
- Use the exact product name from the text. Do not invent names.
- The description is one complete sentence saying what the product is or
  does, written from facts in the text only. Never reuse wording from these
  instructions.
- "evidence" must be copied word for word from the text and must contain
  the product name and the launch / approval / introduction wording.
- If there is no new product, set "announces_new_product": false and
  "products": []."""


# JSON schema passed to Ollama's structured-output mode. Plain "format": "json"
# lets small models emit endless whitespace; a schema plus num_predict cannot.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "announces_new_product": {"type": "boolean"},
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "new_product": {"type": "string"},
                    "product_description": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["new_product", "product_description", "evidence"],
            },
        },
    },
    "required": ["company_name", "announces_new_product", "products"],
}


def query_llm(model, user_prompt):
    """send one chat request to Ollama and return (parsed_json, seconds)."""
    start = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_prompt}],
        "format": RESPONSE_SCHEMA,            # constrain output to the schema
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 700},
    }, timeout=300)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try:
        return json.loads(content), time.time() - start
    except json.JSONDecodeError:
        return {}, time.time() - start


# note: I also tried adding COMPANY and THIRD_PARTY categories here to catch
# acquired companies and other firms' products. llama3.2 (3B) then labelled
# many of the companies' own products as THIRD_PARTY, so those categories were
# removed; such cases are handled in manual_exclusions.csv instead.
VERIFY_PROMPT = """You check candidate products extracted from an SEC 8-K filing.

Company: {company}
Candidate: {name}
Description: {description}
Evidence: {evidence}

Classify the candidate into exactly one category:
- PRODUCT: a product, service, drug, device, software, card, account, app or
  other offering that customers can buy or use
- DEAL: an acquisition, divestiture, joint venture, spin-off, partnership or
  contract
- PROGRAM: an internal or strategic program, financial target or capital plan
- FACILITY: a plant, power plant, data center, store, lounge or other site
- OTHER: a regulator, customer, event, research project or anything else

Return JSON: {{"category": "<one category>", "reason": "<short reason>"}}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string",
                                "enum": ["PRODUCT", "DEAL", "PROGRAM",
                                         "FACILITY", "OTHER"]},
                   "reason": {"type": "string"}},
    "required": ["category", "reason"],
}


def verify_product(model, company, product, evidence=""):
    """
    second LLM pass: classify one candidate; only PRODUCT is kept.
    Newness is already enforced by the evidence check in validate_products,
    so this pass only asks what kind of thing the candidate is, which a small
    model answers far more reliably than "is this new?".
    Returns (ok, reason).
    """
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [{"role": "user", "content": VERIFY_PROMPT.format(
            company=company, name=product["new_product"],
            description=product["product_description"],
            evidence=evidence or "(not available)")}],
        "format": VERIFY_SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 120},
    }, timeout=300)
    resp.raise_for_status()
    try:
        answer = json.loads(resp.json()["message"]["content"])
    except json.JSONDecodeError:
        return False, "unparseable answer"
    category = str(answer.get("category", "OTHER"))
    return category == "PRODUCT", f"{category}: {answer.get('reason', '')}"


def build_prompt(company, ticker, passage, orgs, products):
    """user prompt: filing metadata, NER hints and the selected passages."""
    return (f"Filing company: {company} (ticker {ticker})\n"
            f"NER ORGANIZATION entities: {', '.join(orgs) or 'none'}\n"
            f"NER PRODUCT entities: {', '.join(products) or 'none'}\n\n"
            f"8-K text:\n\"\"\"{passage}\"\"\"\n\n"
            "List the new products announced in this filing.")


# ---------------------------------------------------------------------------
# step 7: validation and cleaning
# ---------------------------------------------------------------------------

def clean_description(text, limit=MAX_DESCRIPTION_CHARS):
    """trim whitespace and cut at a word boundary so len(text) < limit."""
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip('"')
    if len(text) < limit:
        return text
    cut = text[:limit - 4].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "..."


def pretty_company(name):
    """SEC titles are often upper case ("MICROSOFT CORP") -> "Microsoft Corp"."""
    if name.upper() != name:
        return name
    small = {"AND", "OF", "THE"}
    words = [w.lower() if w in small else w.capitalize() for w in name.split()]
    return " ".join(words).replace("& co", "& Co")


def normalise(s):
    """lower-case and keep only letters/digits, so 'Libre® 3' == 'libre 3'."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def strip_symbols(text, source_text):
    """remove trademark marks; llama3.2 sometimes turns '®' into 'µ'."""
    text = re.sub(r"[®™©]", "", text)
    if "µ" not in source_text:
        text = text.replace("µ", "")
    return re.sub(r"\s+", " ", text).strip()


def word_coverage(needle, haystack):
    """share of the words in `needle` that occur in the normalised haystack."""
    words = normalise(needle).split()
    return sum(w in haystack for w in words) / max(len(words), 1)


# names the model returns that are not products (securities, deals, generic
# phrases). Checked against whole words of the normalised name.
NOT_PRODUCT_WORDS = re.compile(
    r"\b(shares|notes|debentures|dividend|transaction|acquisition|merger|"
    r"agreement|webcast|youtube|earnings|solutions?|technologies|"
    r"partnership|campus|facility|products|product mix)\b")


def validate_products(llm_json, source_text):
    """
    keep only grounded, specific products. A product is dropped when
      - its name (or 75% of its words) does not occur in the filing,
      - the name is generic or describes a deal/security instead of a product,
      - the evidence sentence is not (mostly) copied from the filing, does not
        mention the product, or has no launch / approval wording,
      - the description is empty or just repeats the name.
    """
    generic = {"product", "products", "new product", "service", "platform",
               "ai agents", "none", "n a", "unknown", ""}
    haystack = normalise(source_text)
    seen, rows = set(), []
    if llm_json.get("announces_new_product") is False:
        return rows
    for item in llm_json.get("products", []) or []:
        if not isinstance(item, dict):
            continue
        name = strip_symbols(str(item.get("new_product", "")), source_text)
        key = normalise(name)

        # check 1: skip empty/generic names, repeats and run-on sentences.
        if key in generic or key in seen or len(name) > 80:
            continue
        # check 2: skip names that are really deals, securities, etc.
        if NOT_PRODUCT_WORDS.search(key):
            continue
        # check 3: the product name has to appear in the filing (catches
        # names the model made up).
        if key not in haystack and word_coverage(name, haystack) < 0.75:
            continue

        # check 4: the quoted evidence must come from the filing, mention the
        # product, and use a launch word ("launched", "approved", ...).
        evidence = str(item.get("evidence", ""))
        if (word_coverage(evidence, haystack) < 0.8
                or word_coverage(name, normalise(evidence)) < 0.5
                or not LAUNCH_KEYWORDS.search(evidence)):
            continue

        # check 5: the description must add something beyond the name.
        description = clean_description(
            strip_symbols(str(item.get("product_description", "")), source_text))
        if len(description) < 15 or normalise(description) in (key, "the " + key):
            continue

        seen.add(key)
        rows.append({"new_product": name, "product_description": description,
                     "evidence": evidence})
    return rows


# ---------------------------------------------------------------------------
# output files
# ---------------------------------------------------------------------------

def append_csv(path, columns, row):
    new = not path.exists()
    # utf-8-sig writes a BOM on a new file so Excel shows ®, ² and quotes.
    with path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, quoting=csv.QUOTE_ALL)
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_pipe_table(csv_path, txt_path):
    """write the CSV as the ' | ' separated table the assignment shows."""
    rows = read_csv(csv_path)
    headers = ["Company Name", "Stock Name", "Filing Time",
               "New Product", "Product Description"]
    table = [headers] + [[r[c] for c in CSV_COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(4)]

    def fmt(row):
        return " | ".join([row[i].ljust(widths[i]) for i in range(4)] + [row[4]])
    lines = [fmt(headers), "-" * (sum(widths) + 3 * 4 + 20)]
    lines += [fmt(row) for row in table[1:]]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    p.add_argument("--target-rows", type=int, default=100,
                   help="stop once the CSV has this many product rows")
    p.add_argument("--filings-per-company", type=int, default=20,
                   help="how many recent 8-Ks to list per company")
    p.add_argument("--max-docs-per-company", type=int, default=4,
                   help="max filings per company sent to the LLM")
    p.add_argument("--tickers", nargs="*",
                   help="explicit tickers instead of the S&P 500 list")
    p.add_argument("--verify-existing", action="store_true",
                   help="re-check rows already in the CSV with the second LLM "
                        "pass before continuing (used after adding that pass)")
    return p.parse_args()


def apply_manual_exclusions(out_csv):
    """
    remove rows listed in manual_exclusions.csv (stock_name, new_product,
    reason). These are errors found by reading the final CSV by hand that both
    LLM passes missed. Keeping them in a file makes every manual change
    visible and repeatable instead of editing the CSV directly.
    """
    exclusions_file = BASE_DIR / "manual_exclusions.csv"
    if not exclusions_file.exists() or not out_csv.exists():
        return 0
    excluded = {(r["stock_name"], normalise(r["new_product"]))
                for r in read_csv(exclusions_file)}
    rows = read_csv(out_csv)
    kept = [r for r in rows
            if (r["stock_name"], normalise(r["new_product"])) not in excluded]
    if len(kept) != len(rows):
        out_csv.unlink()
        for row in kept:
            append_csv(out_csv, CSV_COLUMNS, row)
    return len(rows) - len(kept)


def check_candidate(model, company, product):
    """verify_product with one retry; an Ollama error counts as a rejection."""
    for attempt in range(2):
        try:
            return verify_product(model, company, product,
                                  product.get("evidence", ""))
        except requests.HTTPError as exc:
            if attempt:
                return False, f"verify_error: {exc}"
            time.sleep(5)


def verify_saved_rows(model, out_csv, rejected_csv):
    """run the second LLM pass over the CSV and rewrite it with the survivors."""
    kept = []
    for row in read_csv(out_csv):
        ok, reason = check_candidate(model, row["company_name"], row)
        print(f"[verify] {row['stock_name']} {row['new_product']!r}: "
              f"{'keep' if ok else 'drop'} ({reason})", flush=True)
        if ok:
            kept.append(row)
        else:
            append_csv(rejected_csv, REJECTED_COLUMNS, {
                "stock_name": row["stock_name"], "filing_time": row["filing_time"],
                "new_product": row["new_product"],
                "product_description": row["product_description"],
                "reason": reason})
    out_csv.unlink()
    for row in kept:
        append_csv(out_csv, CSV_COLUMNS, row)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = OUTPUT_DIR / "new_products.csv"
    log_csv = OUTPUT_DIR / "processing_log.csv"
    rejected_csv = OUTPUT_DIR / "rejected_candidates.csv"

    if args.verify_existing and out_csv.exists():
        verify_saved_rows(args.model, out_csv, rejected_csv)

    # resume support: skip filings already processed in an earlier run.
    done = {row["accession"] for row in read_csv(log_csv)}
    saved = read_csv(out_csv)
    n_rows = len(saved)
    # A product is reported once per company: quarterly releases keep
    # mentioning the same launch, so repeats from other filings are dropped.
    # keyed on the SEC name so dual share classes (FOX / FOXA) count once.
    reported = {(r["company_name"], normalise(r["new_product"])) for r in saved}

    ticker_map = load_ticker_map()
    tickers = args.tickers or load_sp500_tickers()
    print(f"{len(tickers)} companies, model={args.model}, "
          f"{n_rows} rows already saved, {len(done)} filings already seen")

    # outer loop: one company at a time, until we have enough products.
    for ticker in tickers:
        if n_rows >= args.target_rows:
            break
        if ticker.upper() not in ticker_map:
            print(f"[{ticker}] not in SEC ticker file, skipped")
            continue
        cik, company = ticker_map[ticker.upper()]
        company = pretty_company(company)

        try:
            filings = list_8k_filings(cik, args.filings_per_company)
        except requests.RequestException as exc:
            print(f"[{ticker}] feed error: {exc}")
            continue

        # inner loop: this company's 8-Ks, newest first. Every filing gets a
        # row in processing_log.csv saying what happened to it.
        sent_to_llm = 0
        for filing in filings:
            if sent_to_llm >= args.max_docs_per_company or n_rows >= args.target_rows:
                break
            # skip filings done in an earlier run, and amendments (8-K/A).
            if filing["accession"] in done or filing["form"] != "8-K":
                continue
            log = {"ticker": ticker, "company_name": company,
                   "accession": filing["accession"],
                   "filing_time": filing["filing_date"],
                   "items": filing["items"], "doc_url": "", "status": "",
                   "n_products": 0, "ner_orgs": "", "ner_products": "",
                   "llm_seconds": ""}

            # filter 1 (free): only 8-Ks that usually contain a press release.
            if not any(item in filing["items"] for item in RELEVANT_ITEMS):
                log["status"] = "skipped_items"
                append_csv(log_csv, LOG_COLUMNS, log)
                continue

            try:
                text, log["doc_url"] = fetch_filing_text(filing)
            except requests.RequestException as exc:
                log["status"] = f"download_error: {exc}"
                append_csv(log_csv, LOG_COLUMNS, log)
                continue

            # filter 2 (free): the text must contain at least one launch word.
            passage = select_launch_passages(text)
            if not passage:
                log["status"] = "no_launch_keywords"
                append_csv(log_csv, LOG_COLUMNS, log)
                continue

            # NER hints, then the main LLM call (stage 1: extraction).
            orgs, ner_products = run_ner(passage)
            log["ner_orgs"] = "; ".join(orgs)
            log["ner_products"] = "; ".join(ner_products)

            prompt = build_prompt(company, ticker, passage, orgs, ner_products)
            try:
                try:
                    answer, seconds = query_llm(args.model, prompt)
                except requests.HTTPError:
                    time.sleep(5)                 # transient Ollama 500: retry once
                    answer, seconds = query_llm(args.model, prompt)
            except requests.ConnectionError as exc:
                sys.exit(f"Ollama request failed ({exc}). Is `ollama serve` running "
                         f"and `ollama pull {args.model}` done?")
            except requests.RequestException as exc:
                # one bad filing should not stop the run; log it and move on.
                log["status"] = f"llm_error: {exc}"
                append_csv(log_csv, LOG_COLUMNS, log)
                print(f"[{ticker}] {filing['filing_date']} LLM error, skipped", flush=True)
                continue
            sent_to_llm += 1
            log["llm_seconds"] = f"{seconds:.1f}"

            # rule checks on the answer, and drop products we already have.
            candidates = [p for p in validate_products(answer, text)
                          if (company, normalise(p["new_product"])) not in reported]
            # stage 2: a short second LLM call classifies each candidate;
            # anything that is not a PRODUCT goes to rejected_candidates.csv.
            products = []
            for product in candidates:
                ok, reason = check_candidate(args.model, company, product)
                if ok:
                    products.append(product)
                else:
                    append_csv(rejected_csv, REJECTED_COLUMNS, {
                        "stock_name": ticker, "filing_time": filing["filing_date"],
                        "new_product": product["new_product"],
                        "product_description": product["product_description"],
                        "reason": reason})
            reported.update((company, normalise(p["new_product"])) for p in products)
            log["n_products"] = len(products)
            log["status"] = "product_found" if products else "no_product"
            append_csv(log_csv, LOG_COLUMNS, log)

            # save the accepted products right away, so nothing is lost if the
            # run is stopped.
            for product in products:
                append_csv(out_csv, CSV_COLUMNS, {
                    "company_name": company,
                    "stock_name": ticker,
                    "filing_time": filing["filing_date"],
                    "new_product": product["new_product"],
                    "product_description": product["product_description"],
                })
                n_rows += 1
            names = ", ".join(p["new_product"] for p in products) or "-"
            print(f"[{ticker}] {filing['filing_date']} items={filing['items']!r} "
                  f"{seconds:.0f}s -> {names}   (total rows {n_rows})", flush=True)

    # last step: drop the rows flagged during manual review, then write the
    # " | " separated version of the table.
    removed = apply_manual_exclusions(out_csv)
    if removed:
        print(f"Removed {removed} rows listed in manual_exclusions.csv")
    if out_csv.exists():
        write_pipe_table(out_csv, OUTPUT_DIR / "new_products.txt")
    print(f"Done: {len(read_csv(out_csv))} product rows in {out_csv}")


if __name__ == "__main__":
    main()
