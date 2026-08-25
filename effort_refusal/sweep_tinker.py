"""Collect Inkling completions across effort levels, through Tinker.

This is the preferred path, and it is meaningfully stronger than the OpenRouter
one in `sweep.py`.

Why. Through a third-party chat endpoint, effort has to be reconstructed as a
system message and you are left hoping the serving stack does not apply its own
template over the top. That is what the manipulation check in `analyze.py`
exists to catch. Through Tinker, none of that applies: the prompt is built by
Thinking Machines' own `TmlV0Renderer`, and effort is passed as the renderer's
own `effort` argument:

    renderer.build_generation_prompt(messages, effort=0.99)

So the thing being measured is the real code path, not a reconstruction of it.
It is also the exact call site the accompanying `tinker-cookbook` patch is
about: `eval/inspect_utils.py:324` makes this same call without the `effort`
argument, which is why the eval harness cannot reach the operating point the
published numbers use.

Output schema is identical to `sweep.py`, so `classify.py` and `analyze.py`
work unchanged on either.

    pip install tinker tinker-cookbook torch tml-renderers
    set TINKER_API_KEY=...
    python -m effort_refusal.sweep_tinker --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .prompts import load_all, stratified_sample, describe

# tinker_cookbook/renderers/tml_v0.py:68 -> DEFAULT_EFFORT = 0.9
COOKBOOK_DEFAULT_EFFORT = 0.9
# "All evals are run at effort 0.99 and temperature 1.0" - Inkling launch post
PUBLISHED_EFFORT = 0.99

# Thinking Machines' own named effort levels, from
# tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/
#   none 0.0 | minimal 0.1 | low 0.2 | medium 0.7 | high 0.9 (default) | xhigh 0.99
# Sweeping their named stops rather than round numbers of my own choosing means
# every point on the curve is an operating point they actually ship.
EFFORT_LEVELS = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": COOKBOOK_DEFAULT_EFFORT,
    "xhigh": PUBLISHED_EFFORT,
}
DEFAULT_EFFORTS = tuple(EFFORT_LEVELS.values())

# Serverless inference (beta) IDs. These are NOT the training IDs: the docs are
# explicit that the "Tinker ID" column is the exact string for
# create_sampling_client(base_model=...), and the sampling variants carry the
# `:peft:262144:sampling-nvfp4` suffix.
INKLING_SMALL = "thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4"
INKLING = "thinkingmachines/Inkling:peft:262144:sampling-nvfp4"
DEFAULT_MODEL = INKLING_SMALL

# USD per million tokens, from tinker-docs Serverless Inference table.
# Machine-readable source: tinker-docs.thinkingmachines.ai/tinker/serverless.json
PRICES = {
    INKLING_SMALL: {"in": 0.30, "out": 1.20},
    INKLING: {"in": 1.00, "out": 4.05},
}


def build_renderer(sampling_client, renderer_name: str):
    """Build the real TML renderer against the model's own tokenizer.

    The tokenizer comes from the sampling client rather than a local download,
    which keeps this honest: the tokens sent are the tokens the service expects.
    """
    from tinker_cookbook.renderers import get_renderer

    tokenizer = sampling_client.get_tokenizer()
    return get_renderer(name=renderer_name, tokenizer=tokenizer)


def extract(renderer, response, prompt_token_count: int) -> dict:
    """Pull answer text and length signals out of a Tinker SampleResponse."""
    sequence = response.sequences[0]
    tokens = list(sequence.tokens)

    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    try:
        from tinker_cookbook import renderers as _renderers

        for message in renderer.parse_response(tokens):
            text = _renderers.get_text_content(message) or ""
            # TML separates the reasoning trace from the answer. Only the answer
            # is ever labelled: a model that reasons about refusing and then
            # complies has complied.
            if str(message.get("role", "")).lower() in ("thinking", "analysis"):
                thinking_parts.append(text)
            elif message.get("thinking"):
                thinking_parts.append(str(message["thinking"]))
                answer_parts.append(text)
            else:
                answer_parts.append(text)
    except Exception as exc:  # noqa: BLE001
        return {
            "content": "",
            "reasoning": "",
            "parse_error": f"{type(exc).__name__}: {exc}",
            "completion_tokens": len(tokens),
            "prompt_tokens": prompt_token_count,
            "finish_reason": None,
        }

    return {
        "content": "\n".join(p for p in answer_parts if p).strip(),
        "reasoning": "\n".join(p for p in thinking_parts if p).strip(),
        # Total generated tokens is the length signal for the manipulation
        # check, and unlike a third-party endpoint it is always present here.
        "completion_tokens": len(tokens),
        "prompt_tokens": prompt_token_count,
        "finish_reason": getattr(sequence, "stop_reason", None),
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
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"serverless sampling id. small={INKLING_SMALL} "
                             f"flagship={INKLING}")
    parser.add_argument("--renderer", default="tml_v0")
    parser.add_argument("--system-prompt", default=None,
                        help="Optional system prompt. Their playground ships a ~1kB "
                             "one, and system prompts move refusal behaviour, so the "
                             "bare and with-prompt curves are different measurements. "
                             "Pass a file path or a literal string.")
    parser.add_argument("--max-usd", type=float, default=6.0,
                        help="stop if estimated spend exceeds this")
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--efforts", type=float, nargs="+", default=list(DEFAULT_EFFORTS))
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Generous on purpose. A high-effort answer truncated "
                             "mid-thought reads as non-refusal to a string matcher, "
                             "and truncation gets likelier as effort rises, which "
                             "would manufacture the very trend being measured.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Thinking Machines report their evals at temperature 1.0")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="stop after N calls")
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args(argv)

    if not os.environ.get("TINKER_API_KEY"):
        print("set TINKER_API_KEY first", file=sys.stderr)
        return 2

    try:
        import tinker
    except ImportError:
        print("pip install tinker tinker-cookbook torch tml-renderers", file=sys.stderr)
        return 2

    print(f"connecting to Tinker, model={args.model}")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.model)
    renderer = build_renderer(sampling_client, args.renderer)
    print(f"renderer={type(renderer).__name__}")

    system_prompt = args.system_prompt
    if system_prompt and Path(system_prompt).exists():
        system_prompt = Path(system_prompt).read_text(encoding="utf-8")
    if system_prompt:
        print(f"system prompt: {len(system_prompt)} chars")

    prompts = stratified_sample(load_all(), args.per_stratum, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)

    jobs = [(p, e) for p in prompts for e in args.efforts if (p.uid, float(e)) not in done]
    if args.limit:
        jobs = jobs[: args.limit]

    print(describe(prompts))
    print(f"efforts={args.efforts}")
    print(f"{len(jobs)} calls to make, {len(done)} already done -> {args.out}")

    errors = truncated = 0
    with args.out.open("a", encoding="utf-8") as sink:
        for index, (prompt, effort) in enumerate(jobs, start=1):
            record = {
                "uid": prompt.uid,
                "effort": float(effort),
                "mode": "tinker_renderer",
                "model": args.model,
                "source": prompt.source,
                "stratum": prompt.stratum,
                "expected": prompt.expected,
                "prompt": prompt.text,
                "has_system_prompt": bool(system_prompt),
            }
            try:
                messages = [{"role": "user", "content": prompt.text}]
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}] + messages
                # The real thing. This is the call tinker-cookbook's eval path
                # makes without the effort argument.
                model_input = renderer.build_generation_prompt(messages, effort=effort)
                sampling_params = tinker.SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop=renderer.get_stop_sequences(),
                )
                response = sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                ).result()
                record.update(extract(renderer, response, len(model_input.to_ints())))
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"

            sink.write(json.dumps(record) + "\n")
            sink.flush()

            if record.get("error"):
                errors += 1
            if record.get("finish_reason") == "length":
                truncated += 1

            if index % 10 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}  {errors} errors, {truncated} truncated")

            if errors >= 5 and index == errors:
                print(
                    f"\nFirst {errors} calls all failed. Last error:\n"
                    f"  {record.get('error')}\nStopping.",
                    file=sys.stderr,
                )
                return 1

            time.sleep(args.sleep)

    print(f"done -> {args.out}")
    print(f"{errors} errors, {truncated} truncated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
