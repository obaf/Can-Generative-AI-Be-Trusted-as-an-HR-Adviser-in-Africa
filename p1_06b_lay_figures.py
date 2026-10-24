"""Plain-language figures for the Paper 1 lay summary (percentages, large fonts, no jargon).

Reads results/key_numbers.json and results/tables/T_country_accuracy.csv (written by p1_06_analysis.py).
Writes figures/lay_fig1_headline.png and figures/lay_fig2_countries.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import FIGURES, RESULTS

K = json.loads((RESULTS / "key_numbers.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "Calibri", "font.size": 12})

# Figure L1: headline accuracy + confidence vs correctness
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1.35, 1]})
rows = [("10 high-income countries\n(e.g. USA, UK, Germany)", K["acc_High-income comparison"][0], "#7F8C8D"),
        ("10 other developing countries\n(e.g. India, Brazil, Viet Nam)", K["acc_Other LMIC/UMIC comparison"][0], "#A9B7C0"),
        ("54 African countries", K["acc_Africa"][0], "#C0392B")]
for i, (lab, v, c) in enumerate(rows):
    a1.barh(i, v * 100, color=c, height=0.62)
    a1.text(v * 100 + 1.5, i, f"{v * 100:.0f} out of 100", va="center", fontsize=12, fontweight="bold")
a1.set_yticks(range(3), [r[0] for r in rows])
a1.set_xlim(0, 100)
a1.set_xlabel("Questions answered correctly (out of 100)")
a1.set_title("A. How often the chatbots were right", loc="left", fontsize=13, fontweight="bold")

cal = K["calib_Africa"]
vals = [("How sure the\nchatbots said\nthey were", cal["mean_conf"], "#E67E22"), ("How often\nthey were\nactually right", cal["acc"], "#C0392B")]
for i, (lab, v, c) in enumerate(vals):
    a2.bar(i, v * 100, color=c, width=0.58)
    a2.text(i, v * 100 + 2, f"{v * 100:.0f}%", ha="center", fontsize=14, fontweight="bold")
a2.set_xticks(range(2), [v[0] for v in vals])
a2.set_ylim(0, 100)
a2.set_ylabel("%")
a2.set_title("B. African questions: sure vs right", loc="left", fontsize=13, fontweight="bold")
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "lay_fig1_headline.png", dpi=200)
plt.close(fig)

# Figure L2: every African country, in percent
cty = pd.read_csv(RESULTS / "tables" / "T_country_accuracy.csv")
afr = cty[cty.group == "Africa"].sort_values("correct")
hi = cty[cty.group == "High-income comparison"].correct.mean()
pal = {"en": "#2E75B6", "fr": "#8E44AD", "ar": "#16A085", "pt": "#D35400", "es": "#7F8C8D"}
fig, ax = plt.subplots(figsize=(7.5, 11))
ax.barh(afr.economy, afr.correct * 100, color=afr.prompt_lang.map(pal))
ax.axvline(hi * 100, color="black", ls="--", lw=1)
ax.axvline(50, color="#999999", ls=":", lw=1)
ax.set_ylim(-0.8, len(afr) + 0.6)
ax.text(hi * 100, len(afr) - 0.35, f"average for high-income countries ({hi:.0%})", fontsize=9, ha="center", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xlabel("Questions answered correctly (%), average of nine chatbots")
ax.set_xlim(0, 85)
ax.tick_params(axis="y", labelsize=8.5)
ax.legend(title="Main language of the country's labour law",
          handles=[Patch(color=pal[k], label=l) for k, l in [("en", "English (alone or with another language)"), ("fr", "French"), ("ar", "Arabic"),
                                                             ("pt", "Portuguese"), ("es", "Spanish")]],
          fontsize=9, title_fontsize=9, loc="lower right", frameon=True, facecolor="white", edgecolor="white", framealpha=1)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIGURES / "lay_fig2_countries.png", dpi=200)
print("wrote lay figures")
