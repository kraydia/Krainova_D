# Extracting New Product Announcements from SEC 8-K Filings with a Local LLM

**GitHub:** https://github.com/kraydia/Krainova_D

## 1. Goal and output

The task was to read SEC Form 8-K filings, find newly announced products and
save them as `company_name, stock_name, filing_time, new_product,
product_description` (description under 180 characters). The final dataset,
`output/new_products.csv`, has **221 products from 95 S&P 500 companies**.
The pipeline examined **4,930 8-K filings from 495 S&P 500 companies**, filed
between November 2023 and October 2026. The same table, separated by `|`, is
in `output/new_products.txt`.

## 2. Data collection

1. **Company universe:** the S&P 500 constituent table on Wikipedia (503 tickers),
   processed in alphabetical order.
2. **Ticker to CIK:** `https://www.sec.gov/files/company_tickers.json`.
3. **8-K list:** the EDGAR Atom feed
   `browse-edgar?action=getcompany&CIK={cik}&type=8-K&count=20&output=atom`,
   parsed with BeautifulSoup. Each entry gives the accession number, the
   filing date (used as `filing_time`) and the 8-K item numbers.
4. **Item filter:** product launches are announced in press releases. These are
   furnished under Item 7.01 (Reg FD), Item 8.01 (Other Events) or Item 2.02
   (earnings releases, which often list the quarter's launches). Other filings,
   such as executive changes and shareholder votes, are skipped without an LLM
   call. At most four filings per company go to the LLM.
5. **Document text:** from each filing index page the script downloads the
   press-release exhibits (EX-99.x), falling back to the 8-K body if there are
   none, and turns the HTML into plain text. All SEC requests are rate-limited
   (below 10 per second), carry an identifying User-Agent and are cached on disk.

## 3. Pre-processing and Named Entity Recognition

Earnings releases are long, often 20,000 or more characters, while a 3B model
works best with short prompts. The script splits the text into sentences and
keeps every sentence containing a launch keyword (*launch, introduce, unveil,
announce, approval, now available, general availability…*), along with the
sentences on either side and the release headline. This cuts the prompt to at
most 6,000 characters. Filings with no launch keyword never reach the LLM.

**NER:** spaCy's `en_core_web_sm` pipeline tags `ORG` and `PRODUCT` entities in
those passages, and the most frequent ones are sent to the LLM as hints. The
small spaCy model alone is not reliable for products. On the assignment's own
example it labels *Apple Inc.* as ORG correctly but tags *MacBook Pro* as
PERSON and *M3* as GPE. That is why the LLM does the final entity tagging. Its
prompt defines the two entity types, ORGANIZATION and PRODUCT, and asks for
both. Company names in the CSV come from the SEC registrant name, which is
authoritative, and the ticker comes from the SEC mapping.

## 4. LLM extraction

* **Model:** Meta **LLaMA 3.2 (3B parameters, Q4 quantised, 2.0 GB)** served by
  **Ollama 0.40** on a laptop CPU (AMD Ryzen AI 7 445, 23 GB RAM, no
  dedicated GPU). Settings were temperature 0, a 4,096-token context and at most 700 output tokens.
* **Structured output:** the request passes a JSON schema to Ollama
  (`company_name`, `announces_new_product`, and a list of `new_product`,
  `product_description` and `evidence`), so every answer parses.
* **Prompt rules:** a product must be a specific named offering that *this*
  filing says is launched, introduced, approved or made available. Financial
  results, securities, deals, executive changes, business segments and
  existing products are excluded. The model must quote the announcing sentence
  as `evidence`.
* **Rule-based validation**, applied after each answer:
  - the product name must occur in the filing;
  - the evidence must be copied from the filing (at least 80% word overlap),
    contain the product name and contain a launch word;
  - generic names ("solutions", "AI agents") and deal or security words
    ("shares", "acquisition", "webcast") are rejected;
  - the description must say more than the name, and is cut at a word
    boundary to under 180 characters;
  - trademark symbols are removed, and each product is reported once per company.
* **Second LLM pass (verification):** each surviving candidate goes back to the
  model in a short prompt. The model classifies it as PRODUCT, DEAL, PROGRAM,
  FACILITY or OTHER, and only PRODUCT is kept. Rejected candidates and the
  model's reasons are saved in `output/rejected_candidates.csv`.

## 5. Scale and run time

| Stage | Count |
|---|---|
| S&P 500 companies processed (8 had no usable SEC feed entry) | 495 |
| 8-K filings examined | 4,930 |
| skipped: no press-release item (7.01 / 8.01 / 2.02) | 1,494 |
| skipped: no launch keyword in the text | 545 |
| sent to the LLM (stage 1, extraction) | 2,889 |
| unique candidates passing rule-based validation | 281 |
| rejected by the LLM verification pass | 53 (35 PROGRAM, 11 FACILITY, 5 DEAL, 2 OTHER) |
| removed by generic-name rules / duplicate share classes | 7 |
| **final products** | **221 (95 companies)** |

An extraction call took 15 seconds on average, and a verification call took
about 4 seconds. All the LLM work, spread over several resumable runs, came to
roughly 12.5 hours of CPU time. SEC downloads were cached, so changing the
prompt only meant repeating the LLM calls.

**Accuracy:** I read all 221 final rows. About 20 are wrong or doubtful
(around 90% precision). Examples: an acquisition the verifier still accepted
(Ecolab–CoolIT Systems), a spin-off (FedEx Freight), a third-party model
mentioned in passing (Dell–"DeepSeek"), and established products named in an
earnings release (Intel vPro). Recall was not measured: there is no labelled
ground truth, and launches described without any launch keyword are missed.

## 6. Challenges and how they were solved

1. **Endless output in JSON mode.** With Ollama's plain `format: "json"`, the 3B
   model sometimes produced whitespace until the 10-minute timeout, which
   froze the first test run. *Fix:* pass a full JSON schema and cap the output
   at `num_predict`.
2. **Hallucinations and copied text.** In the first full run (110 rows), the
   model copied a sample description from the prompt word for word into
   unrelated drugs. It also listed a YouTube webcast link, preferred shares
   and acquisitions as "products". *Fix:* remove example wording from the
   prompt, require a quoted `evidence` sentence, and check that the evidence
   really occurs in the filing with a launch word.
3. **Existing products in earnings releases.** Item 2.02 releases name many
   established products. The evidence rule (the sentence must say *launched /
   approved / introduced*) removed most of them, and de-duplication stops the
   same launch being counted again in later quarters.
4. **Choosing the verification question.** My first second-pass prompt asked
   "is this a *new* product?". The 3B model answered "established product"
   for almost everything, including AirPods Pro 3 and MacBook Neo, and kept
   24 of 292 candidates. Replacing the yes/no question with a five-way
   category (PRODUCT, DEAL, PROGRAM, FACILITY, OTHER) gave sensible results.
   Small models are much better at classifying what something *is* than at
   judging when it was first released. Because every rejected candidate had been logged,
   I could re-verify the whole candidate pool without repeating extraction
   (`tools/build_pool.py`, then `--verify-existing`).
5. **Weak NER for products.** spaCy's small model misses most product names
   (see section 3), so it is used only to give the LLM hints.
6. **Encoding and symbols.** The model sometimes changed "®" into "µ".
   Trademark symbols are removed, and the CSV is written as UTF-8 with a BOM
   so Excel shows characters such as "²" and "’" correctly.
7. **Run time and robustness.** CPU inference meant runs of several hours.
   The script resumes from its log, retries a transient Ollama 500 error once
   and logs failed filings without stopping.

## 7. Limitations and possible improvements

* A 3B model on a CPU is a trade-off between speed and accuracy. A larger
  model such as `llama3.1:8b` would follow the rules better, but on this
  laptop it would take about three times longer per filing.
* `filing_time` is the EDGAR filing date. The exact acceptance timestamp is
  also available in the feed if finer resolution is needed.
* Earnings releases mix new and existing products. Even with evidence and a
  second pass, a few rows are borderline: a new indication for an existing
  drug, or a new model year of an existing product.
* The keyword filter can miss launches described without any launch word.

## 8. Files

| File | Purpose |
|---|---|
| `extract_8k_products.py` | the full pipeline (commented, command-line options) |
| `output/new_products.csv` | final dataset |
| `output/new_products.txt` | pipe-separated table |
| `output/processing_log.csv` | every filing examined, its status, NER entities and LLM time |
| `output/rejected_candidates.csv` | candidates removed by the second LLM pass |
| `tools/build_pool.py`, `tools/apply_rules.py` | one-off helpers used to re-verify and clean the saved rows (challenge 4) |
| `README.md`, `requirements.txt` | setup and run instructions |
