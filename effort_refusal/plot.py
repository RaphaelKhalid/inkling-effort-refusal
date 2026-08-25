"""Draw the safety-versus-effort curve.

Two error rates on one axis, plotted against Inkling's own named effort stops.
Both are failures, both are percentages, so they share a scale honestly. They
are never averaged into a single "safety score": the 2026 refusal audit (arXiv
2605.05427, 21 models, 7.1M responses) found the two nearly uncorrelated at
r = -0.032, so a combined number would hide the trade this chart exists to show.

    python -m effort_refusal.plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analyze import TRUNCATED, load, wilson
from .classify import is_refusal

# Categorical slots 1 and 2 of the reference palette. Validated for this chart:
# CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the light surface.
BLUE_LIGHT, ORANGE_LIGHT = "#2a78d6", "#eb6834"
BLUE_DARK, ORANGE_DARK = "#3987e5", "#d95926"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#8a8a85", "grid": "#e5e4e0",
        "over": BLUE_LIGHT, "under": ORANGE_LIGHT,
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#8a8a85", "grid": "#333331",
        "over": BLUE_DARK, "under": ORANGE_DARK,
    },
}

# Their own named stops, from the thinking-effort docs page.
LEVEL_ORDER = [("none", 0.0), ("low", 0.2), ("medium", 0.7),
               ("high", 0.9), ("xhigh", 0.99)]


def rates(records: list[dict], labeller: str = "strmatch") -> dict:
    """Per effort: over-refusal on benign, compliance on harmful, with CIs."""
    over: dict[float, list[bool]] = defaultdict(list)
    under: dict[float, list[bool]] = defaultdict(list)

    for record in records:
        # Truncated answers never reached a verdict; counting them would bias
        # the curve toward whichever levels finish early.
        if record.get("finish_reason") in TRUNCATED:
            continue
        refused = is_refusal(record.get(labeller))
        if refused is None:
            continue
        if record["expected"] == "comply":
            over[record["effort"]].append(refused)
        else:
            under[record["effort"]].append(not refused)

    out = {}
    for _, effort in LEVEL_ORDER:
        o, u = over.get(effort, []), under.get(effort, [])
        out[effort] = {
            "over": wilson(sum(o), len(o)) if o else None,
            "under": wilson(sum(u), len(u)) if u else None,
            "n_over": len(o), "n_under": len(u),
        }
    return out


def draw(data: dict, theme_name: str, out_path: Path, model: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    xs = [effort for _, effort in LEVEL_ORDER]

    series = {}
    for key, color, label in (
        ("over", t["over"], "Over-refusal on benign prompts"),
        ("under", t["under"], "Compliance with harmful prompts"),
    ):
        pts = [(x, data[x][key]) for x in xs if data[x][key] is not None]
        if not pts:
            continue
        px = [p for p, _ in pts]
        mid = [v[0] * 100 for _, v in pts]
        lo = [v[1] * 100 for _, v in pts]
        hi = [v[2] * 100 for _, v in pts]
        series[key] = (px, mid, color, label)

        ax.fill_between(px, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(px, mid, color=color, linewidth=2, zorder=3)
        # >= 8px markers, 2px surface ring so overlapping points stay readable
        ax.plot(px, mid, "o", color=color, markersize=8, zorder=4,
                markeredgecolor=t["surface"], markeredgewidth=2)

    # Direct-label at the leftmost point, anchored rightward. The right edge is
    # where the effort annotations live and where the x tick labels crowd in, so
    # labels there collide with furniture rather than with each other. Each label
    # sits above its own line; only if the two lines are close at that x does the
    # lower one flip underneath.
    if series:
        first = min(px[0] for px, _, _, _ in series.values())
        ys = {key: mid[px.index(first)] for key, (px, mid, _, _) in series.items()
              if first in px}
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        crowded = (len(ys) == 2 and abs(list(ys.values())[0] - list(ys.values())[1])
                   < span * 0.12)
        lower = min(ys, key=ys.get) if ys else None

        for key, (px, mid, color, label) in series.items():
            if first not in px:
                continue
            y = mid[px.index(first)]
            dy = -13 if (crowded and key == lower) else 13
            ax.annotate(label, (first, y), textcoords="offset points",
                        xytext=(10, dy), ha="left",
                        va="bottom" if dy > 0 else "top",
                        fontsize=9.5, color=color, zorder=5)

    # The two operating points that carry the argument.
    for effort, note in ((0.9, "their default"), (0.99, "published evals")):
        ax.axvline(effort, color=t["muted"], linewidth=1,
                   linestyle=(0, (2, 3)), zorder=1, alpha=0.7)
        ax.annotate(note, (effort, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(-4, -10), ha="right", va="top",
                    fontsize=8.5, color=t["muted"], rotation=90)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{name}\n{effort}" for name, effort in LEVEL_ORDER],
                       fontsize=9, color=t["secondary"])
    ax.set_xlabel("reasoning effort", fontsize=10, color=t["secondary"], labelpad=8)
    ax.set_ylabel("error rate (%)", fontsize=10, color=t["secondary"], labelpad=8)
    _top = ax.get_ylim()[1]
    ax.set_ylim(0, _top * 1.18)
    ax.set_xlim(-0.07, 1.06)

    ax.set_title(f"Refusal behaviour across effort  ·  {model}",
                 fontsize=12.5, color=t["primary"], loc="left", pad=14)

    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0)

    # Legend is always present for >= 2 series, even though both are labelled.
    ax.legend(
        handles=[
            plt.Line2D([], [], color=t["over"], lw=2, marker="o", markersize=7,
                       label="Over-refusal (benign)"),
            plt.Line2D([], [], color=t["under"], lw=2, marker="o", markersize=7,
                       label="Harmful compliance"),
        ],
        loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2,
        frameon=False, fontsize=9.5, labelcolor=t["secondary"],
    )

    fig.text(0.005, 0.015,
             "Shaded band: 95% Wilson interval. Both series are error rates; "
             "lower is better. Never averaged.",
             fontsize=8, color=t["muted"])

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", type=Path,
                        default=Path("results/labelled.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--labeller", default="strmatch", choices=("strmatch", "judge"))
    args = parser.parse_args(argv)

    records = load(args.src)
    if not records:
        print("no usable records")
        return 1

    model = records[0].get("model", "Inkling")
    data = rates(records, args.labeller)

    print(f"{'level':8} {'effort':>7} {'over-refusal':>16} {'harmful compliance':>20}")
    for name, effort in LEVEL_ORDER:
        d = data[effort]
        o = f"{d['over'][0]:.1%} (n={d['n_over']})" if d["over"] else "-"
        u = f"{d['under'][0]:.1%} (n={d['n_under']})" if d["under"] else "-"
        print(f"{name:8} {effort:>7} {o:>16} {u:>20}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "-dark"
        draw(data, theme, args.out_dir / f"effort-refusal{suffix}.png", model)

    (args.out_dir / "curve.json").write_text(
        json.dumps({str(k): v for k, v in data.items()}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
