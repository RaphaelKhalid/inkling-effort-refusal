"""Collect Inkling completions across reasoning-effort levels.

Why effort is sent as a system message
--------------------------------------
Inkling's effort control is not a bespoke API field. The public `tml-renderers`
wheel renders it as a plain system message:

    <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>

(verified with `tml_renderers.v0.Renderer.render_for_completion_with_effort`;
see `tests/test_effort_rendering.py`, which reproduces it offline). So a sweep
does not depend on the serving endpoint exposing a `reasoning_effort` field.
Any chat endpoint that accepts a system message can carry it.

The manipulation check
----------------------
Sending the directive is not the same as the model honouring it, and a hosted
endpoint may apply its own template on top. So every run records completion and
reasoning token counts per effort level, and `analyze.py` refuses to report a
refusal curve unless reasoning length rises with effort. If the directive is
being swallowed, the token curve is flat and the whole run is void. Better to
find that out from the data than to assume it away.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .prompts import Prompt, load_all, stratified_sample, describe

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The `:free` variant is gated to registered agentic apps and returns HTTP 403
# to a plain script, so this uses the paid endpoint. At $0.95 in / $4.05 out per
# million, a full 600-call sweep costs roughly $2 to $3.
DEFAULT_MODEL = "thinkingmachines/inkling"

# Rough per-million USD, only used for the running cost estimate printed during
# the run. Check openrouter.ai for current numbers.
PRICE_IN_PER_M = 0.95
PRICE_OUT_PER_M = 4.05

# tinker_cookbook/renderers/tml_v0.py:68 -> DEFAULT_EFFORT = 0.9
COOKBOOK_DEFAULT_EFFORT = 0.9
# thinkingmachines.ai/model-card/inkling/ -> table header "Inkling effort=0.99"
PUBLISHED_EFFORT = 0.99

DEFAULT_EFFORTS = (0.0, 0.3, 0.6, COOKBOOK_DEFAULT_EFFORT, PUBLISHED_EFFORT)

EFFORT_DIRECTIVE = "Thinking effort level: {effort}"


def format_effort(effort: float) -> str:
    """Match the renderer's own formatting: 0.0 renders as "0", 0.9 as "0.9"."""
    if effort == int(effort):
        return str(int(effort))
    return str(round(effort, 3))


def build_messages(prompt: Prompt, effort: float, mode: str) -> list[dict]:
    user = {"role": "user", "content": prompt.text}
    if mode == "system":
        directive = EFFORT_DIRECTIVE.format(effort=format_effort(effort))
        return [{"role": "system", "content": directive}, user]
    return [user]


class RateLimited(Exception):
    pass


def call_openrouter(
    messages: list[dict],
    *,
    api_key: str,
    model: str,
    effort: float,
    mode: str,
    max_tokens: int,
    temperature: float,
    timeout: int = 180,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if mode == "param":
        # Fallback path: let the endpoint map effort itself, if it supports it.
        body["reasoning"] = {"effort": effort}
    else:
        # Ask for the reasoning trace to come back. The manipulation check needs
        # a length signal, and without this some providers return only the
        # answer, leaving no way to tell whether the effort directive landed.
        body["reasoning"] = {"enabled": True}

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "inkling-effort-refusal",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 502, 503, 529):
            raise RateLimited(f"HTTP {exc.code}") from exc
        detail = exc.read().decode()[:400]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def extract(response: dict) -> dict:
    """Pull content, reasoning, and token counts out of an OpenRouter reply."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "content": message.get("content") or "",
        # Reasoning models return the trace separately; keep it for the
        # manipulation check but never for the refusal label.
        "reasoning": message.get("reasoning") or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


def load_done(path: Path) -> set[tuple[str, float]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("error"):
                continue
            done.add((record["uid"], float(record["effort"])))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/sweep.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--per-stratum", type=int, default=5,
                        help="prompts per XSTest type / StrongREJECT category")
    parser.add_argument("--efforts", type=float, nargs="+", default=list(DEFAULT_EFFORTS))
    parser.add_argument("--mode", choices=("system", "param", "none"), default="system",
                        help="how to convey effort; 'system' is the verified path")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Generous on purpose. A high-effort answer truncated "
                             "mid-thought looks like non-refusal to a string matcher, "
                             "which would bias the curve in exactly the direction "
                             "being measured. Truncated responses are excluded and "
                             "counted separately in analyze.py.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Thinking Machines report their evals at temperature 1.0")
    parser.add_argument("--max-usd", type=float, default=8.0,
                        help="stop the run if the estimated spend exceeds this")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="seconds between calls")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="stop after N calls (smoke test)")
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("set OPENROUTER_API_KEY (free key at https://openrouter.ai/keys)", file=sys.stderr)
        return 2

    prompts = stratified_sample(load_all(), args.per_stratum, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)

    jobs = [(p, e) for p in prompts for e in args.efforts if (p.uid, float(e)) not in done]
    if args.limit:
        jobs = jobs[: args.limit]

    print(describe(prompts))
    print(f"model={args.model} mode={args.mode} efforts={args.efforts}")
    print(f"{len(jobs)} calls to make, {len(done)} already done -> {args.out}")

    rng = random.Random(args.seed)
    spent_in = spent_out = errors = truncated = 0
    with args.out.open("a", encoding="utf-8") as sink:
        for index, (prompt, effort) in enumerate(jobs, start=1):
            messages = build_messages(prompt, effort, args.mode)
            record = {
                "uid": prompt.uid,
                "effort": float(effort),
                "mode": args.mode,
                "model": args.model,
                "source": prompt.source,
                "stratum": prompt.stratum,
                "expected": prompt.expected,
                "prompt": prompt.text,
            }
            delay = args.sleep
            for attempt in range(args.max_retries):
                try:
                    response = call_openrouter(
                        messages,
                        api_key=api_key,
                        model=args.model,
                        effort=effort,
                        mode=args.mode,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    record.update(extract(response))
                    break
                except RateLimited as exc:
                    backoff = delay * (2 ** attempt) + rng.uniform(0, 1)
                    print(f"  rate limited ({exc}), sleeping {backoff:.0f}s", file=sys.stderr)
                    time.sleep(backoff)
                except Exception as exc:  # noqa: BLE001 - record and move on
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    break
            else:
                record["error"] = "rate limited past max retries"

            sink.write(json.dumps(record) + "\n")
            sink.flush()

            spent_in += record.get("prompt_tokens") or 0
            spent_out += record.get("completion_tokens") or 0
            if record.get("error"):
                errors += 1
            if record.get("finish_reason") == "length":
                truncated += 1

            cost = spent_in / 1e6 * PRICE_IN_PER_M + spent_out / 1e6 * PRICE_OUT_PER_M
            if index % 10 == 0 or index == len(jobs):
                print(
                    f"  {index}/{len(jobs)}  ~${cost:.2f} spent"
                    f"  {errors} errors, {truncated} truncated"
                )

            # Fail loudly and early rather than burning the whole budget on a
            # misconfiguration. Five straight errors means something systemic.
            if errors >= 5 and index == errors:
                print(
                    f"\nFirst {errors} calls all failed. Last error:\n"
                    f"  {record.get('error')}\n"
                    "Stopping before this costs anything more.",
                    file=sys.stderr,
                )
                return 1

            if cost > args.max_usd:
                print(
                    f"\nEstimated spend ${cost:.2f} passed --max-usd {args.max_usd}. "
                    f"Stopping. Rerun to resume; finished work is skipped.",
                    file=sys.stderr,
                )
                return 1

            time.sleep(args.sleep)

    cost = spent_in / 1e6 * PRICE_IN_PER_M + spent_out / 1e6 * PRICE_OUT_PER_M
    print(f"done -> {args.out}")
    print(f"~${cost:.2f} spent, {errors} errors, {truncated} truncated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
