"""Paper 1 — step 6: statistical analysis, tables and figures.

Inputs: results/scored.csv (from p1_05_score.py), data/economies.csv, data/ground_truth.csv
Outputs: results/tables/*.csv, figures/*.png, results/key_numbers.json (numbers quoted in the paper)
"""
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from common import DATA, FIGURES, RESULTS

warnings.filterwarnings("ignore")
TAB = RESULTS / "tables"
TAB.mkdir(exist_ok=True)
GROUP_ORDER = ["Africa", "Other LMIC/UMIC comparison", "High-income comparison"]
GROUP_SHORT = {"Africa": "Africa (54)", "Other LMIC/UMIC comparison": "Other developing (10)", "High-income comparison": "High-income (10)"}
COL = {"Africa": "#C0392B", "Other LMIC/UMIC comparison": "#E0A526", "High-income comparison": "#2E75B6"}
PROTECTIVE_YES = ["WRK_GENDER_EMP", "WRK_AGE_REC", "WRK_MAR_REC", "WRK_FLEX_TIME", "PAY_EQUAL_VALUE", "PAY_NIGHT",
                  "PAY_HAZARD", "PAY_MINING", "PAR_PREG_DISMISS", "SAF_HARASS_EMP", "PEN_EQUAL_AGE"]
BENEFIT_NUMERIC = ["PAR_MAT_DAYS", "PAR_PAT_DAYS", "EPX_NOTICE4", "EPX_REDUND4"]
K = {}


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return lo, hi


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf) / 100, np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


FEDERAL = ["USA", "CAN", "AUS", "DEU", "IND", "PAK", "BRA", "MEX", "NGA", "ETH", "SOM", "SSD", "SDN", "COM"]


def gap_and_or(df):
    """Accuracy by group and the Africa-vs-high-income odds ratio (item + model FE, economy-clustered SE)."""
    d = df.copy()
    d["africa"] = (d.group == "Africa").astype(int)
    d["other_dev"] = (d.group == "Other LMIC/UMIC comparison").astype(int)
    m = smf.glm("correct ~ africa + other_dev + C(item_id) + C(label)", data=d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.iso3)[0]})
    ci = np.exp(m.conf_int().loc["africa"])
    acc = d.groupby("group").correct.mean()
    return dict(n=len(d), acc_africa=acc.get("Africa"), acc_other=acc.get("Other LMIC/UMIC comparison"),
                acc_hi=acc.get("High-income comparison"), OR_africa=np.exp(m.params["africa"]), ci_lo=ci[0], ci_hi=ci[1],
                p=m.pvalues["africa"])


def robustness(s_all):
    en_all = s_all[(s_all.lang == "en") & (s_all.set == "main")]
    rows = {
        "Primary (sub-national rows excluded)": en_all[~en_all.subnational],
        "R1 Full sample (incl. sub-national rows)": en_all,
        "R2 Excluding federal economies": en_all[~en_all.subnational & ~en_all.iso3.isin(FEDERAL)],
        "R3 Lenient numeric tolerance": en_all[~en_all.subnational].assign(correct=lambda x: x.correct_lenient),
        "R4 Non-abstained answers only": en_all[~en_all.subnational & ~en_all.abstain],
        "R5 WBL items only (law as of Oct 2025)": en_all[~en_all.subnational & (en_all.source == "WBL 2026")],
        "R6 Binary items only": en_all[~en_all.subnational & (en_all.type == "binary")],
    }
    t = pd.DataFrame({k: gap_and_or(v) for k, v in rows.items()}).T
    t.round(4).to_csv(TAB / "T13_robustness.csv")
    K["robustness"] = t.round(4).to_dict("index")


def main():
    s_all = pd.read_csv(RESULTS / "scored.csv", encoding="utf-8", low_memory=False)
    robustness(s_all)
    s = s_all[~s_all.subnational].copy()  # primary analysis: national-law ground truth only
    K["n_subnational_rows_excluded_per_model"] = int(s_all[(s_all.set == "main") & (s_all.lang == "en")].groupby("label").subnational.sum().max())
    main_labels = s.loc[s.set == "main", "label"].unique().tolist()
    en = s[(s.lang == "en") & (s.set == "main")].copy()
    K["n_models_main"] = len(main_labels)
    K["n_answers_en_main"] = int(len(en))
    K["n_items_gt"] = int(pd.read_csv(DATA / "ground_truth.csv").shape[0])
    K["n_economies"] = int(en.iso3.nunique())

    # ---------------- T2 accuracy by model × group
    rows = []
    for (lab, g), x in en.groupby(["label", "group"]):
        k, n = x.correct.sum(), len(x)
        lo, hi = wilson(k, n)
        rows.append(dict(model=lab, group=g, n=n, accuracy=k / n, ci_lo=lo, ci_hi=hi, lenient=x.correct_lenient.mean(),
                         abstain=x.abstain.mean()))
    t2 = pd.DataFrame(rows)
    piv = t2.pivot(index="model", columns="group", values="accuracy")[GROUP_ORDER]
    piv["gap_HI_minus_Africa"] = piv["High-income comparison"] - piv["Africa"]
    piv["gap_Other_minus_Africa"] = piv["Other LMIC/UMIC comparison"] - piv["Africa"]
    piv = piv.sort_values("Africa", ascending=False)
    t2.to_csv(TAB / "T2_accuracy_model_group.csv", index=False)
    piv.round(4).to_csv(TAB / "T2b_accuracy_pivot.csv")
    pooled = en.groupby("group").correct.agg(["sum", "size"])
    for g, r in pooled.iterrows():
        lo, hi = wilson(r["sum"], r["size"])
        K[f"acc_{g}"] = [round(r["sum"] / r["size"], 4), round(lo, 4), round(hi, 4), int(r["size"])]
    K["best_model_africa"] = [piv.index[0], round(piv.iloc[0]["Africa"], 4)]
    K["worst_model_africa"] = [piv.index[-1], round(piv.iloc[-1]["Africa"], 4)]
    K["gap_range_HI_minus_Africa"] = [round(piv.gap_HI_minus_Africa.min(), 4), round(piv.gap_HI_minus_Africa.max(), 4)]
    K["models_with_positive_gap"] = int((piv.gap_HI_minus_Africa > 0).sum())

    # Figure 1: dot plot by model
    fig, ax = plt.subplots(figsize=(8, 5.2))
    order = piv.index.tolist()
    for j, g in enumerate(GROUP_ORDER):
        x = t2[t2.group == g].set_index("model").loc[order]
        y = np.arange(len(order)) + (j - 1) * 0.22
        ax.errorbar(x.accuracy, y, xerr=[x.accuracy - x.ci_lo, x.ci_hi - x.accuracy], fmt="o", color=COL[g],
                    label=GROUP_SHORT[g], capsize=2, ms=5)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy on employment-law questions (95% Wilson CI)")
    ax.set_xlim(0.3, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_accuracy_by_model.png", dpi=200)
    plt.close(fig)

    # ---------------- T3 accuracy by item × group
    t3 = en.pivot_table(index=["domain", "item_id"], columns="group", values="correct", aggfunc="mean")[GROUP_ORDER]
    t3["gap_HI_minus_Africa"] = t3["High-income comparison"] - t3["Africa"]
    t3.round(4).to_csv(TAB / "T3_accuracy_item_group.csv")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(t3[GROUP_ORDER].values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], fontsize=8)
    ax.set_yticks(range(len(t3)))
    ax.set_yticklabels([i[1] for i in t3.index], fontsize=7)
    for i in range(len(t3)):
        for j in range(3):
            ax.text(j, i, f"{t3.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_item_heatmap.png", dpi=200)
    plt.close(fig)

    # ---------------- T4 regressions (cluster-robust by economy)
    en["africa"] = (en.group == "Africa").astype(int)
    en["other_dev"] = (en.group == "Other LMIC/UMIC comparison").astype(int)
    m1 = smf.glm("correct ~ africa + other_dev + C(item_id) + C(label)", data=en, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(en.iso3)[0]})
    or1 = np.exp(m1.params[["africa", "other_dev"]])
    ci1 = np.exp(m1.conf_int().loc[["africa", "other_dev"]])
    K["OR_africa_vs_HI"] = [round(or1["africa"], 3), round(ci1.loc["africa", 0], 3), round(ci1.loc["africa", 1], 3),
                            float(m1.pvalues["africa"])]
    K["OR_otherdev_vs_HI"] = [round(or1["other_dev"], 3), round(ci1.loc["other_dev", 0], 3), round(ci1.loc["other_dev", 1], 3),
                              float(m1.pvalues["other_dev"])]
    af = en[en.group == "Africa"].copy()
    af["log_gdp"] = np.log(af.gdp_pc_ppp)
    af["log_pop"] = np.log(af.population)
    af["internet10"] = af.internet_pct / 10
    af["legal"] = af.legal_origin.replace({"English common law": "English", "French civil law": "French"}).fillna("Other")
    af["legal"] = np.where(af.legal.isin(["English", "French"]), af.legal, "Other")
    afc = af.dropna(subset=["log_gdp", "internet10", "log_pop"])
    m2 = smf.glm("correct ~ log_gdp + internet10 + log_pop + C(legal, Treatment('English')) + english_official + C(item_id) + C(label)",
                 data=afc, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(afc.iso3)[0]})
    keep = [p for p in m2.params.index if not p.startswith(("C(item_id)", "C(label)", "Intercept"))]
    t4 = pd.DataFrame({"OR": np.exp(m2.params[keep]), "ci_lo": np.exp(m2.conf_int().loc[keep, 0]),
                       "ci_hi": np.exp(m2.conf_int().loc[keep, 1]), "p": m2.pvalues[keep]})
    t4b = pd.DataFrame({"OR": or1, "ci_lo": ci1[0], "ci_hi": ci1[1], "p": m1.pvalues[["africa", "other_dev"]]})
    pd.concat([t4b.assign(model="M1 all economies"), t4.assign(model="M2 African economies only")]).round(4).to_csv(TAB / "T4_regressions.csv")
    K["M2_n_obs"] = int(m2.nobs)
    K["M2_terms"] = {k: [round(v, 3), round(t4.loc[k, "ci_lo"], 3), round(t4.loc[k, "ci_hi"], 3), round(t4.loc[k, "p"], 4)] for k, v in t4.OR.items()}

    # country-level accuracy (pooled main models, English) + figure 3
    cty = en.groupby(["iso3", "economy", "group", "legal_origin", "prompt_lang", "internet_pct", "gdp_pc_ppp"], dropna=False).correct.mean().reset_index()
    cty.to_csv(TAB / "T_country_accuracy.csv", index=False)
    afr = cty[cty.group == "Africa"].sort_values("correct")
    K["africa_country_acc_range"] = [afr.iloc[0].economy, round(afr.iloc[0].correct, 3), afr.iloc[-1].economy, round(afr.iloc[-1].correct, 3)]
    K["africa_countries_below_50pct"] = int((afr.correct < 0.5).sum())
    fig, ax = plt.subplots(figsize=(7.5, 10))
    colors = afr.prompt_lang.map({"en": "#2E75B6", "fr": "#8E44AD", "ar": "#16A085", "pt": "#D35400", "es": "#7F8C8D"})
    ax.barh(afr.economy, afr.correct, color=colors)
    hi_mean = cty[cty.group == "High-income comparison"].correct.mean()
    ax.axvline(hi_mean, color="black", ls="--", lw=1)
    ax.set_ylim(-0.8, len(afr) + 0.6)
    ax.text(hi_mean, len(afr) - 0.35, f"high-income mean ({hi_mean:.0%})", fontsize=7, ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("Accuracy (pooled across 9 models, English prompts)")
    ax.tick_params(axis="y", labelsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in [("English (alone or bilingual)", "#2E75B6"), ("French", "#8E44AD"),
                                                           ("Arabic", "#16A085"), ("Portuguese", "#D35400"), ("Spanish", "#7F8C8D")]], title="Language of labour law", title_fontsize=7,
              fontsize=7, loc="lower right", frameon=True, facecolor="white", edgecolor="white", framealpha=1)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_country_accuracy.png", dpi=200)
    plt.close(fig)

    # ---------------- T5 calibration
    rows = []
    for (lab, g), x in en[en.confidence.notna() & ~en.abstain].groupby(["label", "group"]):
        y = x.correct.values
        auc = roc_auc_score(y, x.confidence) if 0 < y.mean() < 1 else np.nan
        rows.append(dict(model=lab, group=g, n=len(x), mean_conf=x.confidence.mean() / 100, accuracy=y.mean(),
                         overconfidence=x.confidence.mean() / 100 - y.mean(), ECE=ece(x.confidence, y),
                         brier=np.mean((x.confidence / 100 - y) ** 2), AUROC=auc))
    t5 = pd.DataFrame(rows)
    t5.round(4).to_csv(TAB / "T5_calibration.csv", index=False)
    pc = en[en.confidence.notna() & ~en.abstain]
    for g in GROUP_ORDER:
        x = pc[pc.group == g]
        K[f"calib_{g}"] = dict(mean_conf=round(x.confidence.mean() / 100, 4), acc=round(x.correct.mean(), 4),
                               overconf=round(x.confidence.mean() / 100 - x.correct.mean(), 4), ECE=round(ece(x.confidence, x.correct), 4))
    hc = pc[pc.confidence >= 90]
    K["acc_when_conf_ge_90"] = {g: [round(hc[hc.group == g].correct.mean(), 4), int((hc.group == g).sum())] for g in GROUP_ORDER}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    edges = np.array([0, 50, 60, 70, 80, 90, 95, 101])
    for g in GROUP_ORDER:
        x = pc[pc.group == g]
        b = pd.cut(x.confidence, edges, right=False)
        agg = x.groupby(b, observed=True).agg(conf=("confidence", "mean"), acc=("correct", "mean"), n=("correct", "size"))
        agg = agg[agg.n >= 20]
        ax.plot(agg.conf / 100, agg.acc, "o-", color=COL[g], label=GROUP_SHORT[g])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Observed accuracy")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_calibration.png", dpi=200)
    plt.close(fig)

    # ---------------- T6 statute citations (WBL items with a real legal basis)
    st = en[(en.source == "WBL 2026") & (en.has_statute == True)]
    t6 = st.groupby(["label", "group"]).agg(n=("correct", "size"), cited=("statute_cited", "mean"),
                                           match_given_cited=("statute_match", "mean")).reset_index()
    t6.round(4).to_csv(TAB / "T6_statutes.csv", index=False)
    for g in GROUP_ORDER:
        x = st[st.group == g]
        K[f"statute_{g}"] = dict(cited=round(x.statute_cited.mean(), 4), match=round(x.statute_match.mean(), 4))
    wrongconf = en[(en.correct == 0) & (en.confidence >= 80) & en.statute_cited & ~en.abstain]
    K["confident_wrong_with_citation_share"] = {g: round(len(wrongconf[wrongconf.group == g]) / max((en.group == g).sum(), 1), 4) for g in GROUP_ORDER}

    # ---------------- T7 language condition (paired)
    loc = s[(s.lang != "en") & (s.set == "main")]
    rows = []
    for lab, x in loc.groupby("label"):
        e = s[(s.lang == "en") & (s.label == lab) & s.iso3.isin(x.iso3.unique())][["iso3", "item_id", "correct"]]
        m = x[["iso3", "item_id", "correct"]].merge(e, on=["iso3", "item_id"], suffixes=("_local", "_en"))
        b = ((m.correct_en == 1) & (m.correct_local == 0)).sum()
        c = ((m.correct_en == 0) & (m.correct_local == 1)).sum()
        p = mcnemar([[0, b], [c, 0]], exact=True).pvalue
        rows.append(dict(model=lab, pairs=len(m), acc_en=m.correct_en.mean(), acc_local=m.correct_local.mean(),
                         diff=m.correct_local.mean() - m.correct_en.mean(), en_only=b, local_only=c, mcnemar_p=p))
        for lg, y in x.groupby("lang"):
            my = y[["iso3", "item_id", "correct"]].merge(e, on=["iso3", "item_id"], suffixes=("_local", "_en"))
            rows.append(dict(model=f"{lab} [{lg}]", pairs=len(my), acc_en=my.correct_en.mean(), acc_local=my.correct_local.mean(),
                             diff=my.correct_local.mean() - my.correct_en.mean()))
    t7 = pd.DataFrame(rows)
    t7.round(4).to_csv(TAB / "T7_language.csv", index=False)
    K["language_models"] = t7[~t7.model.str.contains(r"\[")].round(4).to_dict("records")

    # ---------------- T8 frontier subset
    fr_iso = s.loc[s.set == "frontier", "iso3"].unique()
    sub = s[(s.lang == "en") & s.iso3.isin(fr_iso)]
    t8 = sub.pivot_table(index="label", columns="group", values="correct", aggfunc="mean")
    t8["n"] = sub.groupby("label").size()
    t8.round(4).to_csv(TAB / "T8_frontier_subset.csv")
    K["frontier"] = t8.round(4).reset_index().to_dict("records")

    # ---------------- T9 harm direction of errors
    err = en[(en.correct == 0) & ~en.abstain & en.pred.notna()].copy()
    def direction(r):
        if r.item_id in PROTECTIVE_YES:
            return "understates protection" if (r.truth == 1 and r.pred == 0) else "overstates protection"
        if r.item_id in BENEFIT_NUMERIC:
            return "understates entitlement" if r.pred < r.truth else "overstates entitlement"
        return "other"
    err["direction"] = err.apply(direction, axis=1)
    t9 = pd.crosstab(err.group, err.direction, normalize="index")
    t9["n_errors"] = err.groupby("group").size()
    t9.round(4).to_csv(TAB / "T9_error_direction.csv")
    K["error_direction"] = t9.round(4).to_dict()

    # ---------------- T10 anchoring on international defaults (maternity 98/84 days; retirement 60/65)
    mat = en[(en.item_id == "PAR_MAT_DAYS") & en.pred.notna()]
    mat_nd = mat[~mat.truth.isin([98, 84])]
    t10 = mat_nd.assign(anchor=mat_nd.pred.isin([98, 84])).groupby("group").anchor.mean().rename("share_98_or_84_when_truth_differs").to_frame()
    pen = en[en.item_id.isin(["PEN_AGE_MEN", "PEN_AGE_WOMEN"]) & en.pred.notna()]
    pen_nd = pen[~pen.truth.isin([60, 65])]
    t10["share_60_or_65_when_truth_differs"] = pen_nd.assign(a=pen_nd.pred.isin([60, 65])).groupby("group").a.mean()
    t10["maternity_MAPE"] = mat.assign(ape=(mat.pred - mat.truth).abs() / mat.truth.clip(lower=1)).groupby("group").ape.median()
    t10.round(4).to_csv(TAB / "T10_anchoring.csv")
    K["anchoring"] = t10.round(4).to_dict()

    # ---------------- T11 ensemble majority vote + blind spots
    maj = en.groupby(["iso3", "item_id", "group"]).correct.agg(["mean", "size"]).reset_index()
    maj["majority_correct"] = maj["mean"] > 0.5
    maj["all_wrong"] = maj["mean"] == 0
    t11 = maj.groupby("group").agg(majority_acc=("majority_correct", "mean"), share_all_models_wrong=("all_wrong", "mean"),
                                   n_questions=("size", "size"))
    t11.round(4).to_csv(TAB / "T11_ensemble.csv")
    K["ensemble"] = t11.round(4).to_dict()

    # ---------------- T12 abstention / selective accuracy
    t12 = en.groupby("label").agg(abstain_rate=("abstain", "mean"), acc_raw=("correct", "mean"))
    t12["selective_acc"] = en[~en.abstain].groupby("label").correct.mean()
    t12.round(4).to_csv(TAB / "T12_abstention.csv")
    K["abstention"] = t12.round(4).to_dict()

    # ---------------- numeric error sizes by group
    num = en[(en.type == "numeric") & en.pred.notna()]
    K["numeric_median_rel_err"] = num.groupby("group").rel_err.median().round(4).to_dict()
    K["cost_usd_paper1"] = round(float(s.cost.sum()), 3)

    (RESULTS / "key_numbers.json").write_text(json.dumps(K, indent=2, default=str))
    print(json.dumps(K, indent=1, default=str)[:6000])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
