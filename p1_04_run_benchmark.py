"""Paper 1 — step 4: query all models (closed book, temperature 0, one question per call). Resumable.

Runs (in order; budget guard stops the job gracefully if caps are reached):
  A. 9 main models × all English prompts (1,231)
  B. 4 models × official-language prompts for 30 non-anglophone African countries (491)
  C. 2 frontier models × English prompts for a 20-economy subset (~330)
Output: results/raw/<model_slug>.jsonl (one JSON line per call)
"""
import concurrent.futures as cf
import json
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from common import DATA, RESULTS
from llm_client import BudgetExceeded, chat, spent, total_spent

RAW = RESULTS / "raw"
RAW.mkdir(exist_ok=True)
OFF, MIN = {"enabled": False}, {"effort": "minimal"}
MAIN = {  # model -> reasoning setting chosen in the pilot (results/pilot_choice.json)
    "openai/gpt-5.6-luna": OFF, "google/gemini-3.5-flash-lite": MIN, "anthropic/claude-haiku-4.5": OFF,
    "x-ai/grok-4.3": OFF, "deepseek/deepseek-v4-flash": OFF, "meta-llama/llama-4-maverick": OFF,
    "google/gemma-4-31b-it": OFF, "qwen/qwen3.6-35b-a3b": OFF, "mistralai/mistral-small-3.2-24b-instruct": OFF,  # qwen3.8-flash / mistral-small-2603 were persistently rate-limited upstream (429) -> multi-provider open-weight substitutes
}
LANG_MODELS = ["openai/gpt-5.6-luna", "google/gemini-3.5-flash-lite", "deepseek/deepseek-v4-flash", "meta-llama/llama-4-maverick"]
FRONTIER = {"anthropic/claude-sonnet-5": OFF, "openai/gpt-5.6-sol": OFF}
FRONTIER_ECON = ["NGA", "KEN", "ZAF", "GHA", "EGY", "ETH", "MAR", "SEN", "CIV", "TZA", "RWA", "AGO",
                 "USA", "GBR", "FRA", "DEU", "JPN", "IND", "BRA", "IDN"]
_file_locks = {}


def slug(m):
    return m.replace("/", "__")


def done_ids(model):
    p = RAW / f"{slug(model)}.jsonl"
    if not p.exists():
        return set()
    ids = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("text") and not d.get("error"):
            ids.add(d["prompt_id"])
    return ids


def run_one(model, reasoning, row):
    res = chat(model, [{"role": "user", "content": row.prompt}], paper="paper1", max_tokens=300, reasoning=reasoning)
    rec = {"prompt_id": row.prompt_id, "model": model, "lang": row.lang, "text": res["text"], "cost": res["cost"],
           "usage": res["usage"], "latency": res["latency"], "finish_reason": res["finish_reason"],
           "model_returned": res["model_returned"], "provider": res["provider"], "error": res["raw_error"],
           "reasoning_cfg": reasoning, "ts": datetime.now(timezone.utc).isoformat()}
    lock = _file_locks.setdefault(model, threading.Lock())
    with lock, open(RAW / f"{slug(model)}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def run_batch(model, reasoning, df, workers=6):
    todo = df[~df.prompt_id.isin(done_ids(model))]
    print(f"[{time.strftime('%H:%M:%S')}] {model}: {len(todo)} to do (of {len(df)})", flush=True)
    errors = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, model, reasoning, r) for r in todo.itertuples()]
        for f in cf.as_completed(futs):
            try:
                if f.result()["error"]:
                    errors += 1
            except BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                raise
    print(f"[{time.strftime('%H:%M:%S')}] {model}: finished, errors={errors}, paper1 spend={spent('paper1'):.3f}", flush=True)


def main():
    p = pd.read_csv(DATA / "prompts.csv", encoding="utf-8")
    en, other = p[p.lang == "en"], p[p.lang != "en"]
    try:
        # A: main models, English — run models concurrently (each with its own small pool)
        with cf.ThreadPoolExecutor(max_workers=len(MAIN)) as ex:
            list(ex.map(lambda kv: run_batch(kv[0], kv[1], en), MAIN.items()))
        # B: language condition
        with cf.ThreadPoolExecutor(max_workers=len(LANG_MODELS)) as ex:
            list(ex.map(lambda m: run_batch(m, MAIN[m], other), LANG_MODELS))
        # C: frontier subset
        sub = en[en.iso3.isin(FRONTIER_ECON)]
        with cf.ThreadPoolExecutor(max_workers=len(FRONTIER)) as ex:
            list(ex.map(lambda kv: run_batch(kv[0], kv[1], sub), FRONTIER.items()))
    except BudgetExceeded:
        pass
    print("ALL DONE. total spend (both papers):", round(total_spent(), 4), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
