"""Draw docs/architecture.png (used in the README and slides)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, TEAL, GREEN, GOLD, RED, GREY = "#1F3A5F", "#2A7F8E", "#4C7A43", "#C98A1B", "#B04A4A", "#555555"
FILL = {NAVY: "#EEF3F8", TEAL: "#E3F2F3", GREEN: "#E8F0E6", GOLD: "#FFF4E0", RED: "#FBEAEA"}

fig, ax = plt.subplots(figsize=(12, 6.6), dpi=170)
ax.set_xlim(0, 120)
ax.set_ylim(0, 66)
ax.axis("off")


def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=FILL[color], ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=10.5, weight="bold", color="#1B2733")
    ax.text(x + w / 2, y + h - 5.6, sub, ha="center", va="top", fontsize=8.2, color="#1B2733", linespacing=1.35)


def arrow(x1, y1, x2, y2, color=GREY, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color,
                                 lw=1.3, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


# phases
phases = ["1 Clarify goal\n(input guardrails)", "2 Gather evidence\n(retrieval)", "3 Plan + test loop\n(ToT beam search)",
          "4 Validate finalists\n(1,000 questions)", "5 Report\n(cited, advisory)"]
for i, p in enumerate(phases):
    x = 2 + i * 23.6
    c = GOLD if i == 2 else TEAL
    ax.add_patch(FancyBboxPatch((x, 58), 21, 6.4, boxstyle="round,pad=0.2,rounding_size=1", fc=FILL[c], ec=c, lw=1.2))
    ax.text(x + 10.5, 61.2, p, ha="center", va="center", fontsize=8.4, weight="bold", color="#1B2733")
    if i < 4:
        arrow(x + 21.3, 61.2, x + 23.4, 61.2)

box(2, 40, 17, 12, "Engineer", "token cap +\nlatency cap", NAVY)
box(38, 41, 44, 11, "Orchestrator: LangGraph StateGraph", "clarify > evidence > expand > test > evaluate (loop) >\nvalidate > report; budget, pruning; interrupt() for humans", NAVY)
arrow(19.3, 46, 37.7, 46.5, style="<|-|>")

ag = [("Researcher (CrewAI)", "judges research evidence\nvia MCP search_research"), ("Planner (CrewAI)", "writes Thoughts, picks fills;\nwrites the summary"),
      ("Experimenter (tool)", "no LLM: runs BM25 / TF-IDF /\nBGE+FAISS; time limit + retry"), ("Evaluator (CrewAI)", "code: rubric, prune, confidence;\nLLM critic via MCP get_tree")]
xs = [2, 32, 62, 92]
for x, (t, s) in zip(xs, ag):
    box(x, 21, 26, 12.5, t, s, GREEN)
    arrow(60, 40.7, x + 13, 33.8, color="#8A9BB0")
arrow(28.3, 27, 31.7, 27, color=TEAL)
arrow(58.3, 28.5, 61.7, 28.5, color=GOLD)
arrow(88.3, 28.5, 91.7, 28.5, color=GOLD)
arrow(102, 20.7, 48, 20.7, color=GOLD, rad=-0.12)
ax.text(80, 13.2, "scores + prune decisions -> Planner", fontsize=8, color=GOLD, ha="center")

box(2, 1.5, 116, 9, "Shared state: MCP server (search_research, search_experiments, get_tree)",
    "search tree + scores   |   experiment log (JSONL, every run has an ID)   |   research knowledge base (11 notes, injection filter)   |   agents: Claude Haiku 5.5",
    GOLD)
for x in xs:
    arrow(x + 3, 20.7, x + 3, 11.0, color="#C9A35B", style="<|-|>", ls=(0, (2, 2)))

ax.text(102, 46.5, "Human checkpoints", fontsize=9, weight="bold", color=RED, ha="center")
ax.text(102, 44.2, "ambiguous goal, disallowed data,\ninfeasible request, >50% budget step,\nlow confidence (<0.80), final sign-off",
        fontsize=7.8, color=RED, ha="center", va="top", linespacing=1.35)
out = Path(__file__).with_name("architecture.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
