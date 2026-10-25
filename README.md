# New-product extraction from SEC 8-K filings with a local LLM

Pulls recent Form 8-K filings for S&P 500 companies from SEC EDGAR, finds
passages about product launches, tags entities with spaCy NER, and asks a local
LLaMA 3.2 model (via Ollama) to return the new products as structured JSON.
Answers are checked against the filing text, and a second LLM pass keeps only
candidates it classifies as products (not deals, programs or facilities).

Result: **196 new products from 84 companies**, from 4,930 filings examined
(25 rows removed by manual review are listed in `manual_exclusions.csv`).
See [report.md](report.md) for the methodology.

## Output

| File | Contents |
|------|----------|
| `output/new_products.csv` | `company_name, stock_name, filing_time, new_product, product_description` |
| `output/new_products.txt` | the same table, values separated by ` \| ` |
| `output/processing_log.csv` | every filing examined, its status, NER entities and LLM time |
| `output/rejected_candidates.csv` | candidates dropped by the verification pass, with the model's reason |
| `tools/` | one-off helpers used to re-verify and clean the saved rows |
| `report.md` | methodology report |

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.2
set SEC_USER_AGENT=Your Name your.email@example.com
python extract_8k_products.py --target-rows 100
```

Options: `--model` (any Ollama model, e.g. `llama3.1:8b`), `--tickers AAPL NVDA`
(instead of the S&P 500 list), `--filings-per-company`, `--max-docs-per-company`,
`--verify-existing` (re-run the verification pass over rows already saved).
The run resumes where it stopped: filings already in the log are skipped.

## Data sources

- `https://www.sec.gov/files/company_tickers.json` maps ticker to CIK
- `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&count={count}&output=atom` lists a company's 8-Ks
- Wikipedia's *List of S&P 500 companies* provides the company universe
