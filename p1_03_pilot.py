"""Paper 1 — step 3: fast parallel pilot of candidate models × reasoning settings (no retries).

For each model, try reasoning settings in order (off → effort none → minimal → provider default) on one prompt;
keep the first that returns a parseable answer without hidden reasoning tokens where possible, then check
format stability on 3 more prompts. Output: results/pilot.csv and results/pilot_choice.json
"""
import concurrent.futures as cf
import json
import os
import sys
import time

import pandas as pd
import requests

from common import DATA, RESULTS
from llm_client import _add_cost, total_spent

CANDIDATES = [
    "openai/gpt-5.6-luna", "google/gemini-3.5-flash-lite", "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-v4-flash", "qwen/qwen3.8-flash", "meta-llama/llama-4-maverick",
    "mistralai/mistral-small-2603", "google/gemma-4-31b-it", "x-ai/grok-4.3",
    "anthropic/claude-sonnet-5", "openai/gpt-5.6-sol", "google/gemini-3.1-pro-preview",
]
VARIANTS = [("off", {"enabled": False}), ("none", {"effort": "none"}), ("minimal", {"effort": "minimal"}), ("default", None)]
PIDS = ["NGA|PAR_MAT_DAYS|en", "KEN|WRK_AGE_REC|en", "FRA|PAY_NIGHT|en", "GHA|EPX_NOTICE4|en"]


def once(model, prompt, rcfg, timeout=60):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "temperature": 0, "seed": 7, "usage": {"include": True}}
    if rcfg is not None:
        body["reasoning"] = rcfg
    t0 = time.time()
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        d = r.json()
    except Exception as e:
        return dict(ok=False, err=f"{type(e).__name__}", text="", cost=0, rt=None, ct=None, lat=round(time.time() - t0, 1))
    if r.status_code != 200 or "choices" not in d:
        return dict(ok=False, err=f"{r.status_code} {json.dumps(d.get('error', d))[:160]}", text="", cost=0, rt=None, ct=None,
                    lat=round(time.time() - t0, 1))
    u = d.get("usage", {}) or {}
    cost = float(u.get("cost") or 0)
    _add_cost("paper1", cost)
    txt = (d["choices"][0].get("message") or {}).get("content") or ""
    return dict(ok=bool(txt.strip()), err="", text=txt, cost=cost,
                rt=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), ct=u.get("completion_tokens"),
                lat=round(time.time() - t0, 1))


def probe(model, prompts):
    rows, chosen = [], None
    for vname, cfg in VARIANTS:
        res = once(model, prompts[0], cfg)
        rows.append(dict(model=model, variant=vname, prompt_id=PIDS[0], **res))
        if res["ok"]:
            chosen = (vname, cfg)
            break
    if chosen:
        for pid, pr in zip(PIDS[1:], prompts[1:]):
            res = once(model, pr, chosen[1])
            rows.append(dict(model=model, variant=chosen[0], prompt_id=pid, **res))
    return rows, (chosen[0] if chosen else None)


def main():
    p = pd.read_csv(DATA / "prompts.csv", encoding="utf-8").set_index("prompt_id")
    prompts = [p.at[i, "prompt"] for i in PIDS]
    allrows, choice = [], {}
    with cf.ThreadPoolExecutor(max_workers=len(CANDIDATES)) as ex:
        futs = {ex.submit(probe, m, prompts): m for m in CANDIDATES}
        for f in cf.as_completed(futs):
            rows, ch = f.result()
            allrows += rows
            choice[futs[f]] = ch
            last = rows[-1]
            print(futs[f], "->", ch, "| cost", round(sum(r["cost"] for r in rows), 5), "| rt", [r["rt"] for r in rows],
                  "|", (last["text"] or last["err"])[:100].replace("\n", " "), flush=True)
    pd.DataFrame(allrows).to_csv(RESULTS / "pilot.csv", index=False, encoding="utf-8")
    (RESULTS / "pilot_choice.json").write_text(json.dumps(choice, indent=2))
    print("total spend (all papers):", round(total_spent(), 4))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
