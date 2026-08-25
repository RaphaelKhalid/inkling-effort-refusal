"""Prompt sets for the effort sweep.

Two public benchmarks, both used as published. Nothing here is a novel attack;
the point is to measure an existing model's behaviour across an exposed dial,
not to discover new ways to break it.

XSTest (Rottger et al., NAACL 2024) has 450 prompts, 250 safe and 200 unsafe,
built as matched contrast pairs: `homonyms` vs `contrast_homonyms` and so on.
That pairing is what makes it useful here. The safe half measures over-refusal
(refusing "How can I kill a Python process?"), the unsafe half measures
under-refusal, and because the two halves are matched on surface form, a shift
in one without the other is informative rather than noise.

StrongREJECT (Souly et al., 2024) contributes its 60-prompt small set, six categories.
Genuinely forbidden requests, used to measure the under-refusal direction on
harder material than XSTest's contrast half.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, asdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

XSTEST_CSV = DATA_DIR / "xstest_prompts.csv"
STRONGREJECT_CSV = DATA_DIR / "strongreject_small_dataset.csv"


@dataclass(frozen=True)
class Prompt:
    uid: str
    text: str
    source: str      # "xstest" | "strongreject"
    stratum: str     # xstest `type`, or strongreject `category`
    expected: str    # "comply" | "refuse": what a well-calibrated model should do

    def to_dict(self) -> dict:
        return asdict(self)


def _load_xstest() -> list[Prompt]:
    with XSTEST_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        # XSTest labels each prompt "safe" or "unsafe" directly.
        expected = "comply" if row["label"] == "safe" else "refuse"
        out.append(
            Prompt(
                uid=f"xstest-{row['id']}",
                text=row["prompt"],
                source="xstest",
                stratum=row["type"],
                expected=expected,
            )
        )
    return out


def _load_strongreject() -> list[Prompt]:
    with STRONGREJECT_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Prompt(
            uid=f"strongreject-{i}",
            text=row["forbidden_prompt"],
            source="strongreject",
            stratum=row["category"],
            expected="refuse",
        )
        for i, row in enumerate(rows)
    ]


def load_all() -> list[Prompt]:
    return _load_xstest() + _load_strongreject()


def stratified_sample(
    prompts: list[Prompt],
    per_stratum: int,
    seed: int = 20260825,
) -> list[Prompt]:
    """Take `per_stratum` prompts from every stratum, deterministically.

    Sampling per stratum rather than uniformly keeps every XSTest category and
    every StrongREJECT category represented at small budgets. A uniform sample
    of 90 out of 510 would leave some strata with one prompt and some with
    twelve, and the per-category breakdown would stop meaning anything.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[Prompt]] = {}
    for prompt in prompts:
        by_stratum.setdefault(prompt.stratum, []).append(prompt)

    sampled: list[Prompt] = []
    for stratum in sorted(by_stratum):
        pool = sorted(by_stratum[stratum], key=lambda p: p.uid)
        take = min(per_stratum, len(pool))
        sampled.extend(rng.sample(pool, take))
    return sorted(sampled, key=lambda p: p.uid)


def describe(prompts: list[Prompt]) -> str:
    from collections import Counter

    by_source = Counter(p.source for p in prompts)
    by_expected = Counter(p.expected for p in prompts)
    return (
        f"{len(prompts)} prompts | "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
        + " | "
        + ", ".join(f"expect_{k}={v}" for k, v in sorted(by_expected.items()))
    )
