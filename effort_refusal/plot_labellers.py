"""Draw the two published labellers against each other.

This is the figure the project ends on. Both series are the same quantity
(compliance with prompts that should be refused) measured over the same
responses; the only thing that differs is which of XSTest's two published
labellers scored them. Where they diverge, at least one is wrong.

    python -m effort_refusal.plot_labellers --in results/labelled-inkling.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .analyze import TRUNCATED, load, spearman, wilson
from .classify import is_refusal

# Categorical slots 1 and 2 of the reference palette, validated for this pair:
# CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the light surface.
THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#8a8a85", "grid": "#e5e4e0",
              "judge": "#2a78d6", "strmatch": "#eb6834"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#8a8a85", "grid": "#333331",
             "judge": "#3987e5", "strmatch": "#d95926"},
}

LEVELS = [("none", 0.0), ("low", 0.2), ("medium", 0.7), ("high", 0.9), ("xhigh", 0.99)]

SERIES = [
    ("strmatch", "String matcher", "XSTest's published prefix matcher"),
    ("judge", "Rubric judge", "XSTest's published LLM rubric"),
]


def compliance(records: list[dict], labeller: str) -> dict:
    """Compliance rate on prompts that should be refused, per effort level."""
    buckets: dict[float, list[bool]] = defaultdict(list)
    for record in records:
        if record.get("finish_reason") in TRUNCATED:
            continue
        if record["expected"] != "refuse":
            continue
        refused = is_refusal(record.get(labeller))
        if refused is None:
            continue
        buckets[record["effort"]].append(not refused)
    return {e: (wilson(sum(buckets[e]), len(buckets[e])), len(buckets[e]))
            for _, e in LEVELS if buckets.get(e)}


def draw(data: dict, theme_name: str, out: Path, model: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    drawn = {}
    for key, short, _ in SERIES:
        pts = data.get(key) or {}
        xs = [e for _, e in LEVELS if e in pts]
        if not xs:
            continue
        mid = [pts[x][0][0] * 100 for x in xs]
        lo = [pts[x][0][1] * 100 for x in xs]
        hi = [pts[x][0][2] * 100 for x in xs]
        colour = t[key]
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, linewidth=0)
        ax.plot(xs, mid, color=colour, linewidth=2, zorder=3)
        ax.plot(xs, mid, "o", color=colour, markersize=8, zorder=4,
                markeredgecolor=t["surface"], markeredgewidth=2)
        drawn[key] = (xs, mid, colour, short)

    # Label each line above its own first point; the two never converge here.
    for key, (xs, mid, colour, short) in drawn.items():
        ax.annotate(short, (xs[0], mid[0]), textcoords="offset points",
                    xytext=(10, 12), ha="left", va="bottom",
                    fontsize=10, color=colour, zorder=5)

    ax.set_xticks([e for _, e in LEVELS])
    ax.set_xticklabels([f"{n}\n{e}" for n, e in LEVELS], fontsize=9,
                       color=t["secondary"])
    ax.set_xlabel("reasoning effort", fontsize=10, color=t["secondary"], labelpad=8)
    ax.set_ylabel("compliance with harmful prompts (%)", fontsize=10,
                  color=t["secondary"], labelpad=8)
    ax.set_ylim(0, max(ax.get_ylim()[1], 45) * 1.06)
    ax.set_xlim(-0.07, 1.06)

    # Subtitle sits under the title, so the title needs room above the axes.
    ax.set_title("Same responses, two published labellers, opposite conclusions",
                 fontsize=13, color=t["primary"], loc="left", pad=30)
    ax.text(0, 1.035, model, transform=ax.transAxes, fontsize=9.5,
            color=t["muted"], va="bottom")

    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0)

    ax.legend(handles=[plt.Line2D([], [], color=t[k], lw=2, marker="o",
                                  markersize=7, label=s)
                       for k, s, _ in SERIES],
              loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
              frameon=False, fontsize=9.5, labelcolor=t["secondary"])

    fig.text(0.005, 0.015,
             "Shaded band: 95% Wilson interval. Both lines are the same "
             "quantity over the same responses.",
             fontsize=8, color=t["muted"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path,
                        default=Path("results/labelled-inkling.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--name", default="labeller-divergence")
    args = parser.parse_args(argv)

    records = load(args.src)
    if not records:
        print("no usable records")
        return 1
    model = records[0].get("model", "Inkling")
    data = {key: compliance(records, key) for key, _, _ in SERIES}

    print(f"{'level':8}{'effort':>7}" + "".join(f"{s:>18}" for _, s, _ in SERIES))
    for name, e in LEVELS:
        row = f"{name:8}{e:>7}"
        for key, _, _ in SERIES:
            d = data[key].get(e)
            row += f"{(f'{d[0][0]:.1%} (n={d[1]})' if d else '-'):>18}"
        print(row)
    for key, short, _ in SERIES:
        xs = [e for _, e in LEVELS if e in data[key]]
        ys = [data[key][e][0][0] for e in xs]
        print(f"  Spearman(effort, {short}) = {spearman(xs, ys):+.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "-dark"
        draw(data, theme, args.out_dir / f"{args.name}{suffix}.png", model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
