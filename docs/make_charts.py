"""Draw docs/tradeoff.png and docs/ablation.png from the evaluation outputs."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (light)
INK, MUTED, GRID = "#1B2733", "#5c6670", "#e3e6ea"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

grid = list(csv.DictReader((RES / "grid_test.csv").open()))
summary = json.loads((RES / "eval_summary.json").read_text())

# ---------------- 1. trade-off scatter
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
for (ret, color, marker, label) in [("bm25", SERIES[0], "o", "BM25"), ("tfidf", SERIES[1], "s", "TF-IDF"),
                                    ("glove", SERIES[2], "^", "GloVe dense")]:
    pts = [r for r in grid if r["retriever"] == ret]
    ax.scatter([float(r["tokens"]) for r in pts], [float(r["accuracy"]) for r in pts], s=26, c=color,
               marker=marker, edgecolors="white", linewidths=0.8, label=label, zorder=3)
for cap, lab in [(200, "S1 cap"), (400, "S2 cap"), (1000, "S3 / S5 cap")]:
    ax.axvline(cap, color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.text(cap * 1.03, 0.265, lab, color=MUTED, fontsize=7.5)
picks = {}
for name, row in summary["scenarios"].items():
    top = Counter(p["pick"] for p in row["agent"]["per_run"] if p.get("pick")).most_common(1)[0][0]
    picks.setdefault(top, []).append(name.split("_")[0])
by_key = {r["config"]: r for r in grid}
groups = {}
for key, names in picks.items():
    r = by_key[key]
    x, y = float(r["tokens"]), float(r["accuracy"])
    ax.scatter([x], [y], s=110, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
    gk = (round(x, -1), round(y, 1))
    groups.setdefault(gk, [x, y, []])[2].extend(names)
for x, y, names in groups.values():
    ax.annotate(", ".join(sorted(names)), (x, y), xytext=(-11, 0), textcoords="offset points",
                fontsize=8, color=INK, weight="bold", ha="right", va="center")
ax.set_xscale("log")
ax.set_xticks([100, 200, 400, 1000, 2000])
ax.set_xticklabels(["100", "200", "400", "1,000", "2,000"])
ax.set_xlabel("Mean context tokens per question (log scale)")
ax.set_ylabel("Answer recall (test, 2,000 q)")
ax.grid(axis="y", color=GRID, lw=0.7)
ax.set_ylim(0.25, 1.0)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("All 81 configurations on the held-out test split; rings = agent's most frequent pick per scenario",
             fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "tradeoff.png", facecolor="white")

# ---------------- 2. ablation bars (two panels, one measure each)
names = list(summary["scenarios"])
short = [n.split("_")[0] for n in names]
variants = [("agent", "Agent (beam 3)"), ("greedy", "Greedy (beam 1)"), ("no_kb", "No knowledge base")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=200)
w = 0.26
for panel, (metric, title) in zip(axes, [("mean_regret", "Mean regret vs. exhaustive best (lower is better)"),
                                         ("mean_experiments", "Experiments per request (of 81)")]):
    for i, (v, label) in enumerate(variants):
        vals = [summary["scenarios"][n][v][metric] for n in names]
        xs = [j + (i - 1) * (w + 0.02) for j in range(len(names))]
        panel.bar(xs, vals, width=w, color=SERIES[i], label=label, zorder=3)
    panel.set_xticks(range(len(names)))
    panel.set_xticklabels(short)
    panel.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    panel.set_title(title, fontsize=8.5, color=INK, loc="left")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "ablation.png", facecolor="white")
print("ok")
