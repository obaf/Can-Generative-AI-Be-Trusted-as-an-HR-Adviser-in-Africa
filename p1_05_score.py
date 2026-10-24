"""Paper 1 — step 5: parse model answers and score them against the ground truth.

Scoring rules (pre-registered in assumptions.docx):
  * Binary items: correct iff the Yes/No answer equals the WBL coding. "Unsure" = abstention (counted wrong
    in raw accuracy; excluded in selective accuracy).
  * Numeric items: strict tolerance — leave days: max(1 day, 5%); retirement ages: 0.5 years;
    EPLex months: max(0.1 month, 10%). Lenient tolerance — within 25% (or ±1 unit if truth ≤ 4).
  * Units: if the answer string states weeks/months/days in a different unit from the one requested, convert
    (7 days/week, 30.44 days/month, 4.345 weeks/month) and flag the conversion.
  * Statute-citation match (WBL items with a real legal basis): shares a law number/year token with the WBL
    legal basis, or fuzzy title similarity ≥ 80 after translating common French/Portuguese/Spanish keywords.
Output: results/scored.csv, results/models.csv
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from common import DATA, RESULTS

RAW = RESULTS / "raw"
MODELS = pd.DataFrame([
    # id, label, developer, developer country, open weights (HF repo listed by OpenRouter), OpenRouter listing date, run set
    ("openai/gpt-5.6-luna", "GPT-5.6 Luna", "OpenAI", "USA", False, "2026-07-09", "main"),
    ("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "Google", "USA", False, "2026-07-21", "main"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", "Anthropic", "USA", False, "2025-10-15", "main"),
    ("x-ai/grok-4.3", "Grok 4.3", "xAI", "USA", False, "2026-04-30", "main"),
    ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", "DeepSeek", "China", True, "2026-04-24", "main"),
    ("meta-llama/llama-4-maverick", "Llama 4 Maverick", "Meta", "USA", True, "2025-04-05", "main"),
    ("google/gemma-4-31b-it", "Gemma 4 31B", "Google", "USA", True, "2026-04-02", "main"),
    ("qwen/qwen3.6-35b-a3b", "Qwen 3.6 35B-A3B", "Alibaba (Qwen)", "China", True, "2026-04-27", "main"),
    ("mistralai/mistral-small-3.2-24b-instruct", "Mistral Small 3.2 24B", "Mistral AI", "France", True, "2025-06-20", "main"),
    ("anthropic/claude-sonnet-5", "Claude Sonnet 5", "Anthropic", "USA", False, "2026-06-30", "frontier"),
    ("openai/gpt-5.6-sol", "GPT-5.6 Sol", "OpenAI", "USA", False, "2026-07-09", "frontier"),
], columns=["model", "label", "developer", "dev_country", "open_weights", "listed", "set"])

YES = {"yes", "oui", "sim", "si", "sí", "نعم", "true", "y"}
NO = {"no", "non", "não", "nao", "لا", "false", "n"}
UNSURE = {"unsure", "unknown", "incertain", "incerto", "no sé", "غير متأكد", "not sure", "n/a", ""}
TRANSL = {  # keyword translation for statute-title matching
    "code du travail": "labour code", "código do trabalho": "labour code", "codigo do trabalho": "labour code",
    "código del trabajo": "labour code", "lei geral do trabalho": "general labour law", "loi": "law", "lei": "law",
    "ley": "law", "décret": "decree", "decreto": "decree", "ordonnance": "ordinance", "sécurité sociale": "social security",
    "segurança social": "social security", "labor": "labour", "proclamation": "proclamation", "arrêté": "order",
}


def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        blob = m.group()
        for candidate in (blob, re.sub(r",\s*}", "}", blob)):
            try:
                return json.loads(candidate), True
            except ValueError:
                pass
    # regex fallback
    out = {}
    a = re.search(r'"?answer"?\s*:\s*"?([^",}\n]+)', t)
    c = re.search(r'"?confidence"?\s*:\s*"?(\d+(?:\.\d+)?)', t)
    lb = re.search(r'"?legal_basis"?\s*:\s*"([^"]*)"', t)
    if a:
        out["answer"] = a.group(1).strip()
    if c:
        out["confidence"] = float(c.group(1))
    if lb:
        out["legal_basis"] = lb.group(1)
    return out, False


def to_binary(v):
    s = str(v).strip().lower().strip(".")
    if s in YES or s.startswith("yes"):
        return 1.0
    if s in NO or s.startswith("no ") or s == "no":
        return 0.0
    if s in ("1", "1.0"):
        return 1.0
    if s in ("0", "0.0"):
        return 0.0
    return np.nan


def to_number(v, unit):
    """Return (value, converted_flag). Converts stated units to the requested one."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v), False
    s = str(v).strip().lower()
    if s in UNSURE:
        return np.nan, False
    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    if not m:
        return np.nan, False
    x = float(m.group().replace(",", "."))
    conv = False
    if unit == "days":
        if "week" in s or "semaine" in s or "semana" in s:
            x, conv = x * 7, True
        elif "month" in s or "mois" in s or "mes" in s:
            x, conv = x * 30.44, True
    elif unit in ("months", "months_wages"):
        if "week" in s or "semaine" in s or "semana" in s:
            x, conv = x / 4.345, True
        elif "day" in s or "jour" in s or "dia" in s or "día" in s:
            x, conv = x / 30.44, True
        elif "year" in s or "an" == s[-2:]:
            x, conv = x * 12, True
    return x, conv


def tol_strict(item, truth):
    if item.startswith("PAR_"):
        return max(1.0, 0.05 * truth)
    if item.startswith("PEN_"):
        return 0.5
    return max(0.1, 0.10 * truth)


def tol_lenient(truth):
    return 1.0 if truth <= 4 else 0.25 * truth


def statute_tokens(s):
    s = strip_accents(str(s).lower())
    nums = set(re.findall(r"\b\d{1,4}[-/]\d{1,4}\b|\b(?:19|20)\d{2}\b|\bno\.?\s*\d+", s))
    return nums


def title_part(s):
    s = str(s).lower()
    for a, b in TRANSL.items():
        s = s.replace(a, b)
    s = strip_accents(s)
    s = re.split(r",?\s*(?:sec(?:tion)?s?\.?|art(?:icle|s)?\.?|§)\s*\d", s)[0]
    return re.sub(r"[^a-z0-9 ]", " ", s)


def statute_match(model_lb, wbl_lb):
    if not model_lb or str(model_lb).strip().lower() in ("unknown", "", "n/a", "none"):
        return np.nan  # not cited
    parts = [p for p in re.split(r";", str(wbl_lb)) if p.strip()]
    mt = statute_tokens(model_lb)
    for p in parts:
        if mt & statute_tokens(p):
            return 1.0
        if fuzz.token_set_ratio(title_part(model_lb), title_part(p)) >= 80:
            return 1.0
    return 0.0


def load_raw():
    recs = []
    for mid in MODELS.model:
        f = RAW / f"{mid.replace('/', '__')}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("text") and not d.get("error"):
                recs.append(d)
    raw = pd.DataFrame(recs)
    raw = raw.sort_values("ts").groupby(["model", "prompt_id"]).tail(1)  # latest successful answer
    return raw


def main():
    gt = pd.read_csv(DATA / "ground_truth.csv")
    econ = pd.read_csv(DATA / "economies.csv")
    raw = load_raw()
    raw[["iso3", "item_id", "lang"]] = raw.prompt_id.str.split("|", expand=True)
    df = raw.merge(gt, on=["iso3", "item_id"], how="left").merge(
        econ[["iso3", "economy", "group", "region", "income", "legal_origin", "prompt_lang", "english_official",
              "NY.GDP.PCAP.PP.KD", "IT.NET.USER.ZS", "SP.POP.TOTL"]], on="iso3", how="left")
    df = df[df.truth.notna()]  # answers to prompts whose ground truth was later judged ambiguous are not scored
    rows = []
    for r in df.itertuples():
        obj, clean = parse_json(r.text)
        ans = obj.get("answer", "")
        conf = obj.get("confidence", np.nan)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = np.nan
        if conf == conf and conf <= 1.0 and conf > 0:  # some models answer on a 0-1 scale
            conf *= 100
        lb = obj.get("legal_basis", "")
        abstain = str(ans).strip().lower() in UNSURE
        conv = False
        if r.type == "binary":
            pred = np.nan if abstain else to_binary(ans)
            correct = float(pred == r.truth) if pred == pred else 0.0
            lenient, abs_err, rel_err = correct, np.nan, np.nan
        else:
            pred, conv = (np.nan, False) if abstain else to_number(ans, r.unit)
            if pred == pred and r.item_id == "EPX_PROBATION" and r.truth >= 25:
                # EPLex codes "no statutory maximum" as 25 months (sub-indicator AREA2 = 0). The prompt told models to
                # answer 0 "if the law provides none", so 0 (none) or >= 24 months both express the correct rule.
                correct = lenient = float(pred == 0 or pred >= 24)
                abs_err = rel_err = np.nan
            elif pred == pred:
                abs_err = abs(pred - r.truth)
                rel_err = abs_err / max(r.truth, 1.0)
                correct = float(abs_err <= tol_strict(r.item_id, r.truth))
                lenient = float(abs_err <= tol_lenient(r.truth))
            else:
                correct, lenient, abs_err, rel_err = 0.0, 0.0, np.nan, np.nan
        rows.append(dict(model=r.model, prompt_id=r.prompt_id, iso3=r.iso3, item_id=r.item_id, lang=r.lang,
                         type=r.type, unit=r.unit, truth=r.truth, answer_raw=str(ans)[:60], pred=pred,
                         abstain=abstain, parse_ok=bool(obj) and "answer" in obj, clean_json=clean,
                         unit_converted=conv, correct=correct, correct_lenient=lenient, abs_err=abs_err,
                         rel_err=rel_err, confidence=conf, legal_basis_model=str(lb)[:200],
                         statute_cited=not (str(lb).strip().lower() in ("unknown", "", "n/a", "none")),
                         statute_match=statute_match(lb, r.legal_basis) if (r.source == "WBL 2026" and r.has_statute) else np.nan,
                         source=r.source, domain=r.domain, has_statute=r.has_statute, subnational=r.subnational,
                         economy=r.economy, group=r.group,
                         region=r.region, income=r.income, legal_origin=r.legal_origin, prompt_lang=r.prompt_lang,
                         english_official=r.english_official, cost=r.cost, latency=r.latency))
    out = pd.DataFrame(rows)
    # covariates by explicit merge (itertuples renames columns containing dots)
    out = out.merge(
        econ[["iso3", "NY.GDP.PCAP.PP.KD", "IT.NET.USER.ZS", "SP.POP.TOTL"]].rename(
            columns={"NY.GDP.PCAP.PP.KD": "gdp_pc_ppp", "IT.NET.USER.ZS": "internet_pct", "SP.POP.TOTL": "population"}),
        on="iso3", how="left").merge(MODELS, on="model", how="left")
    out.to_csv(RESULTS / "scored.csv", index=False, encoding="utf-8")
    MODELS.to_csv(RESULTS / "models.csv", index=False)
    summ = out.groupby(["label", "lang"] if out.lang.nunique() > 1 else ["label"]).agg(
        n=("correct", "size"), acc=("correct", "mean"), parse_ok=("parse_ok", "mean"), abstain=("abstain", "mean"),
        conf=("confidence", "mean"), statute_match=("statute_match", "mean")).round(3)
    print(summ.to_string())


if __name__ == "__main__":
    main()
