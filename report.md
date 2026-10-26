# Finding New Products in SEC 8-K Filings with a Local LLM

**Daria Krainova** | GitHub: https://github.com/kraydia/Krainova_D

## Goal

The goal was to read SEC Form 8-K filings, find new product announcements,
and save them in a CSV with five columns: company name, ticker, filing date,
product name and a short description (under 180 characters). For "Filing
Time" I used the SEC filing date (YYYY-MM-DD), the same format as the example
in the assignment, rather than the exact time of day. My final file,
`output/new_products.csv`, has **189 products from 83 S&P 500 companies**.
To get there, the program looked at **4,930 8-K filings from 495 companies**,
filed between November 2023 and October 2026.

## How I collected the data

I started with the list of S&P 500 companies from Wikipedia. For each ticker
I looked up the company's CIK number in the SEC file `company_tickers.json`.
Then I used the EDGAR Atom feed
(`browse-edgar?action=getcompany&CIK=...&type=8-K&output=atom`) to get the
company's 20 most recent 8-Ks, and parsed it with BeautifulSoup.

Most 8-Ks are not about products (for example executive changes or
shareholder votes), so I only kept filings under Item 7.01, 8.01 or 2.02.
Those are the items where companies attach press releases. For each one I
downloaded the press release (Exhibit 99) and turned the HTML into plain
text. To save time, I only sent the sentences containing words like
*launch*, *introduce*, *unveil*, *approval* or *now available* to the LLM.

## Named Entity Recognition

I used spaCy (`en_core_web_sm`) to find ORG and PRODUCT entities in the
text, and passed the most common ones to the LLM as hints. spaCy alone was
not good enough for products. On the assignment's own example it tagged
"MacBook Pro" as a PERSON. So the LLM made the final decision, using a prompt
that defines the two entity types (ORGANIZATION and PRODUCT).

## The LLM

- **Model:** Meta LLaMA 3.2, 3B parameters (2 GB), run locally with Ollama
- **Computer:** laptop with an AMD Ryzen AI 7 CPU and 23 GB RAM, no GPU
- **Settings:** temperature 0, so the same input always gives the same answer

The LLM is used in two steps:

1. **Extraction.** The model gets the filtered text and returns JSON with each
   new product, a one-sentence description, and the exact sentence from the
   filing that announces it ("evidence").
2. **Checking.** Each product goes back to the model in a short prompt that
   sorts it into one of five categories: PRODUCT, DEAL, PROGRAM, FACILITY or
   OTHER. Only PRODUCT is kept.

Between the two steps, my code runs some simple checks. The product name
must appear in the filing. The evidence sentence must really be in the
filing and contain a launch word. The description has to say more than just
the product name. Each product is only counted once per company.

## Results

| Step | Count |
|---|---|
| 8-K filings looked at | 4,930 |
| Skipped: wrong item type | 1,494 |
| Skipped: no launch words | 545 |
| Sent to the LLM | 2,889 |
| Products that passed my checks | 281 |
| Removed by the LLM checking step | 53 |
| Removed by name rules / duplicates | 7 |
| Removed by my manual review | 32 |
| **Final products** | **189** |

Each extraction call took about 15 seconds, so all the LLM work added up to
about 12 hours on my laptop. I ran it in several sessions, and the script
picks up where it stopped.

**Manual review:** I read every row of the final file. I removed 32 that were
clearly wrong, such as an acquired company (Ecolab's "CoolIT Systems"), a
spin-off ("FedEx Freight"), another company's AI model (Dell's "DeepSeek"),
existing products (Intel vPro), and drugs or devices that were only in
clinical studies, such as results from Abbott's AVEIR feasibility study or
experimental drugs from a pipeline list. For unclear rows I checked the
original filing text. I didn't edit the CSV by hand. Each removed
row is listed with its reason in `manual_exclusions.csv`, and the script
applies that list at the end. A few remaining rows are borderline, for
example a newly approved use for an existing drug.

## Problems I ran into

1. **The model got stuck.** With Ollama's basic JSON mode the model sometimes
   printed blank space until it timed out. Giving it a strict JSON schema and
   an output limit fixed this.
2. **Made-up answers.** In my first full run, the model copied an example
   sentence from my prompt into the descriptions of unrelated drugs, and
   called things like a YouTube webcast or preferred shares "products". I
   took the example out of the prompt and added the evidence check.
3. **Asking the right question.** My first checking prompt asked "is this
   product new?". The small model answered "no" for almost everything,
   including AirPods Pro 3. Asking it to pick a category worked much better.
   Later I tried adding categories for "another company's product" and "a
   company". That caught DeepSeek and CoolIT, but it also threw out many
   real products, such as Datadog's own AI tools. So I removed those
   categories and handled those cases in the manual review instead.
4. **Earnings releases** list a lot of old products next to new ones. The
   evidence check and de-duplication handle most of these.
5. **Small fixes:** the model sometimes turned "®" into "µ", so I remove
   trademark symbols. Ollama occasionally returned an error, so the script
   retries once and then skips that filing.

## Limitations

- A 3B model is fast but not very accurate. A larger model like
  `llama3.1:8b` would probably follow the rules better, but would be about
  three times slower on my laptop.
- I could only measure how many results were wrong, not how many launches
  were missed. There is no answer key, and launches described without any
  launch word are skipped.
- `filing_time` is the filing date, not the exact time. The EDGAR feed also
  has a full timestamp, so the script could save that instead if needed.

## Files

`extract_8k_products.py` (the program), `output/new_products.csv` (results),
`output/new_products.txt` (same table with `|`), `output/processing_log.csv`
(every filing and what happened to it), `output/rejected_candidates.csv`
(products the LLM check removed), `manual_exclusions.csv` (rows I removed by
hand), `tools/` (helper scripts) and `README.md` (how to run it).
