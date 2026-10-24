# Can Generative AI Be Trusted as an HR Adviser in Africa?

**A 54-country benchmark of large language models on employment law — replication code**

This repository contains the complete, runnable pipeline behind the study *"Can Generative AI Be Trusted as an HR Adviser in Africa? A 54-Country Benchmark of Large Language Models on Employment Law"* (manuscript, 2026). The pipeline downloads expert-coded legal data and turns it into 1,227 employment-law questions for all 54 African countries and 20 comparison economies. It then asks nine widely used large language models (LLMs) every question closed-book, scores each answer against the legal ground truth, and produces every table, figure and number reported in the paper.

**Headline result.** The models answered correctly for 55.1% of African questions, against 60.5% for other developing and 69.3% for high-income economies (odds ratio for Africa vs high-income = 0.46). Every model did worse on African law. The models were strongly overconfident (stated confidence 85% vs accuracy 58%), and their mistakes followed a pattern: they substituted international or conventional defaults for national law.

---

## What is innovative and unique about this work

1. **Continent-wide and multi-model.** This is the first audit of generative-AI legal knowledge for human resource management that covers *every* African country. It spans nine LLMs from eight developers, plus two frontier models, and compares Africa with both high-income and other developing economies. The result is 10,854 scored primary answers, plus 1,964 legal-language and 704 frontier-model answers. Earlier work covered the fifty US states or one African country at a time.
2. **An expert, citable answer key.** Ground truth comes from harmonised, statute-referenced legal indicators: the World Bank's *Women, Business and the Law 2026*, law as of 1 October 2025, and the ILO's EPLex dismissal-law indicators. It does not rely on answer keys written by the researchers. The legal reference date is stated in every prompt. The code specifies how ambiguous codes are handled: partial scores, "no applicable provisions", sub-national coding in federal economies and EPLex's "no statutory maximum" code.
3. **More than right or wrong.** Every answer carries a stated confidence and a cited statute. The pipeline measures calibration: expected calibration error, AUROC of confidence, and accuracy when the model claims at least 90% confidence. It checks whether cited statutes match the official legal basis, and it counts answers that are *confident, wrong and backed by a citation*. These are the signals non-expert users rely on most.
4. **Mechanism, not only magnitude.** The analysis classifies the *direction* of every error: overstating or understating a protection or an entitlement. It also tests for *default substitution*, for example answering a pension age of 60 or 65, or the ILO's 14- or 12-week maternity leave, when national law differs. This underpins an institutional–epistemic account of why models fail where legal information is scarce.
5. **A language experiment.** The same questions are asked in each country's legal language (French, Arabic, Portuguese or Spanish), using grammatically correct country forms, and compared with paired exact McNemar tests. This separates a *language-interface* problem from a *knowledge* problem.
6. **Robustness and a ground-truth audit.** Six robustness specifications test sub-national rows, federal economies, lenient tolerances, abstentions, WBL-only items and binary-only items. Separately, a purposive audit of confident consensus errors against primary legal sources checks the answer key itself.
7. **Built to be reproduced cheaply.** The pipeline uses one API interface (OpenRouter) and a hard budget guard with spending ledgers. Runs are resumable, use temperature 0 and a fixed seed, and store every raw response. The whole benchmark costs about US$2.70 in API fees. Every number in the manuscript is inserted by code from the results files, so no result is typed by hand.

---

## Repository layout

```
code/                     all scripts, run in the order below
  p1_02_build_items.py    build the ground truth and the prompt bank
  p1_03_pilot.py          optional pilot (model availability, JSON compliance, reasoning settings)
  p1_04_run_benchmark.py  query every model (resumable)
  p1_05_score.py          parse and score every answer
  p1_06_analysis.py       regressions, calibration, robustness, tables, figures, key_numbers.json
  p1_06b_lay_figures.py   plain-language figures

```

The scripts create `data/`, `results/` and `figures/` next to `code/` as they run. Large downloads go to a cache directory outside the repository (see step 1).

---

## Requirements

* **Python 3.14** (tested on Windows 11; the code is platform-independent apart from the optional PowerShell helper in `tools/`).
* Python packages: `pip install -r requirements.txt` (pinned to the versions used in the study).
* **An OpenRouter API key** (https://openrouter.ai), about **US$3 of credit** and internet access.
* Optional, only to rebuild the manuscript: **pandoc 3.8** with citeproc.

---

## Reproducing the experiments: step by step

### Step 0 — Get the code and set up the environment

```bash
git clone <this repository> ai-hr-africa-paper1
cd ai-hr-africa-paper1
python -m venv .venv
.venv\Scripts\activate          # Windows;  on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1 — Configure the API key

```bash
# Windows (PowerShell):  $env:OPENROUTER_API_KEY = "sk-or-..."  

```

* `OPENROUTER_API_KEY` is read from the environment and never printed or stored.


### Step 2 — Download the open data (about 1 minute)


Download the WBL 2026 data file, the ILO EPLex 2020 coded indicators and two Quality of Government cross-sections (the January 2022 edition supplies La Porta legal-origin codes).

### Step 3 — Build the ground truth and the prompts (seconds)

```bash
python p1_02_build_items.py
```


### Step 4 — (Optional) pilot the models (about US$0.10)

```bash
python p1_03_pilot.py
```

### Step 5 — Run the benchmark (about US$2.70; 1–2 hours)

```bash
python p1_04_run_benchmark.py
```

* **Phases:** (A) 9 main models × 1,227 English questions; (B) 4 models × 491 legal-language questions; (C) 2 frontier models × a 20-economy subset.
* **Resumable:** if the run stops (rate limits, network errors or the budget cap), run the same command again. Finished prompts are skipped.
* **Settings:** temperature 0, seed 7, at most 300 output tokens, internal reasoning disabled (minimal for Gemini 3.5 Flash-Lite).
* **Models used:** `openai/gpt-5.6-luna`, `google/gemini-3.5-flash-lite`, `anthropic/claude-haiku-4.5`, `x-ai/grok-4.3`, `deepseek/deepseek-v4-flash`, `meta-llama/llama-4-maverick`, `google/gemma-4-31b-it`, `qwen/qwen3.6-35b-a3b` and `mistralai/mistral-small-3.2-24b-instruct`. The frontier models are `anthropic/claude-sonnet-5` and `openai/gpt-5.6-sol`. If a model has been retired, substitute the closest successor in `MAIN` / `FRONTIER` at the top of the script and report the substitution.

### Step 6 — Score and analyse (about 2 minutes, no API calls)

```bash
python p1_05_score.py
python p1_06_analysis.py
python p1_06b_lay_figures.py
```


---


## Data sources and licences

| Source | Used for | Licence / terms |
|---|---|---|
| World Bank, *Women, Business and the Law 2026* | 15 legal items (law as of 1 Oct 2025) | CC BY 4.0 |
| ILO, EPLex coded indicators (2020 edition) | 3 dismissal-law items | free use with attribution (ILO terms) |
| Quality of Government standard cross-section (Jan 2022, Jan 2026) | legal origin | free for academic use with citation |
| World Bank, World Development Indicators | GDP, internet use, population | CC BY 4.0 |
| Wikidata | official languages | CC0 |

---

## How to cite

Please cite the article: *Can Generative AI Be Trusted as an HR Adviser in Africa? A 54-Country Benchmark of Large Language Models on Employment Law* (manuscript under review, 2026). Full reference to be added on publication.
