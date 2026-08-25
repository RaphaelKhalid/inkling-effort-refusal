"""Collect Inkling completions across effort levels, via Tinker's
Anthropic-compatible endpoint.

This is the path to run first. It talks to the endpoint over plain HTTP with
the standard library, so it needs no dependencies at all: no `anthropic` SDK,
no torch, no tml-renderers. That is deliberate. The SDK route broke on a
signature mismatch (`Messages.create() got an unexpected keyword argument
'temperature'`), and an eval harness that only runs on one patch version of one
client library is not much of an eval harness.

Effort is set through the Tinker-specific field their docs prescribe. Through
the SDK that arrives as `extra_body={"output_config": {"effort": "high"}}`;
over raw HTTP it is simply a top-level `output_config` key.

Named levels map to the floats published at
tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/ :

    none 0.0 | minimal 0.1 | low 0.2 | medium 0.7 | high 0.9 (default) | xhigh 0.99

Sweeping their own named stops means every point on the curve is an operating
point they actually ship, rather than round numbers picked by me. `none` is not
accepted by `output_config.effort`, so it is expressed the way their docs
specify instead: `thinking: {"type": "disabled"}`.

Two things this endpoint gives that a third-party one does not: reasoning comes
back as its own `thinking` content block, so the refusal label is computed on
the answer alone; and usage is reported, so the manipulation check has a real
length signal.

    set TINKER_API_KEY=...
    python -m effort_refusal.sweep_anthropic --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from pathlib import Path

from .prompts import load_all, stratified_sample, describe

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api"
MESSAGES_URL = f"{BASE_URL}/v1/messages"

INKLING_SMALL = "thinkingmachines/Inkling-Small"
INKLING = "thinkingmachines/Inkling"
DEFAULT_MODEL = INKLING_SMALL

EFFORT_LEVELS: dict[str, float] = {
    "none": 0.0,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,      # their default, and tinker-cookbook's DEFAULT_EFFORT
    "xhigh": 0.99,    # "All evals are run at effort 0.99" - Inkling launch post
}

# USD per million tokens, serverless inference table.
PRICES = {
    INKLING_SMALL: {"in": 0.30, "out": 1.20},
    INKLING: {"in": 1.00, "out": 4.05},
}


class RateLimited(Exception):
    pass


# Cloudflare sits in front of this endpoint and rejects clients by signature:
# a bare urllib request announces itself as "Python-urllib/3.x" and comes back
# HTTP 403 "error code: 1010". httpx is what the Anthropic SDK itself uses, so
# it is tried first, with requests and urllib behind it. Every transport sends
# a real User-Agent.
USER_AGENT = "inkling-effort-refusal/1.0 (research eval; +https://github.com/RaphaelKhalid)"


def _post(url: str, body: dict, headers: dict, timeout: int) -> tuple[int, str]:
    """POST via the best available transport. Returns (status, text)."""
    payload = json.dumps(body)

    try:
        import httpx
    except ImportError:
        pass
    else:
        try:
            response = httpx.post(url, content=payload, headers=headers, timeout=timeout)
            return response.status_code, response.text
        except Exception as exc:  # noqa: BLE001 - network failure, not HTTP status
            return 0, f"httpx transport error: {type(exc).__name__}: {exc}"

    try:
        import requests
    except ImportError:
        pass
    else:
        try:
            response = requests.post(url, data=payload, headers=headers, timeout=timeout)
            return response.status_code, response.text
        except Exception as exc:  # noqa: BLE001
            return 0, f"requests transport error: {type(exc).__name__}: {exc}"

    request = urllib.request.Request(
        url, data=payload.encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def build_body(
    prompt_text: str,
    level: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None,
) -> dict:
    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if system_prompt:
        body["system"] = system_prompt
    if level == "none":
        # Their docs: thinking.disabled forces reasoning off regardless of
        # output_config, and `none` is not a valid output_config.effort value.
        body["thinking"] = {"type": "disabled"}
        body["temperature"] = temperature
    else:
        body["output_config"] = {"effort": level}
        # Anthropic's API rejects temperature alongside extended thinking, so
        # it is only sent on the no-thinking path. The published evals use
        # temperature 1.0, which is the API default anyway.
    return body


def call(body: dict, *, api_key: str, timeout: int = 300) -> dict:
    headers = {
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": USER_AGENT,
    }
    status, text = _post(MESSAGES_URL, body, headers, timeout)

    if status == 200:
        return json.loads(text)
    if status in (429, 502, 503, 529):
        raise RateLimited(f"HTTP {status}: {text[:300]}")
    if status == 403 and "1010" in text:
        raise RuntimeError(
            "HTTP 403 Cloudflare code 1010: the request was rejected by client "
            "fingerprint, not by your API key. Install httpx and rerun: "
            "pip install httpx"
        )
    raise RuntimeError(f"HTTP {status}: {text[:400]}")


def extract(payload: dict) -> dict:
    """Split the reply into reasoning and answer, and pull usage counts."""
    thinking_parts, answer_parts = [], []
    for block in payload.get("content") or []:
        kind = block.get("type")
        if kind == "thinking":
            thinking_parts.append(block.get("thinking") or "")
        elif kind == "text":
            answer_parts.append(block.get("text") or "")

    usage = payload.get("usage") or {}
    # This endpoint reports input tokens across three fields and routinely
    # returns input_tokens=0 with the real count under cache_creation_input_tokens.
    # Summing all three keeps the cost meter honest.
    prompt_tokens = sum(
        usage.get(key) or 0
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return {
        # Only the answer is ever labelled. A model that reasons about refusing
        # and then complies has complied.
        "content": "\n".join(p for p in answer_parts if p).strip(),
        "reasoning": "\n".join(p for p in thinking_parts if p).strip(),
        "finish_reason": payload.get("stop_reason"),
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": usage.get("output_tokens"),
        "usage_raw": usage,
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


def probe(api_key: str, model: str) -> int:
    """One call, everything printed verbatim. For when something is wrong and
    guessing is expensive."""
    import platform

    print(f"python      {platform.python_version()} on {platform.system()}")
    for name in ("httpx", "requests", "anthropic"):
        try:
            module = __import__(name)
            print(f"{name:11} {getattr(module, '__version__', 'installed')}")
        except ImportError:
            print(f"{name:11} NOT INSTALLED")
    print(f"key         {api_key[:8]}...{api_key[-4:]} ({len(api_key)} chars)")
    print(f"url         {MESSAGES_URL}")
    print(f"model       {model}\n")

    body = {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say OK and nothing else."}],
        "output_config": {"effort": "low"},
    }
    headers = {
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "application/json",
        "user-agent": USER_AGENT,
    }
    print(f"request body:\n{json.dumps(body, indent=2)}\n")
    status, text = _post(MESSAGES_URL, body, headers, 120)
    print(f"HTTP {status}")
    print(text[:2000])
    return 0 if status == 200 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/sweep.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"{INKLING_SMALL} (cheap) or {INKLING} (flagship)")
    parser.add_argument("--levels", nargs="+", default=list(EFFORT_LEVELS),
                        choices=list(EFFORT_LEVELS))
    parser.add_argument("--per-stratum", type=int, default=5,
                        help="prompts per stratum; -1 uses every prompt in both datasets")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="parallel in-flight requests. Serverless inference is "
                             "in beta and their docs discourage high throughput, so "
                             "this stays deliberately modest.")
    parser.add_argument("--system-prompt", default=None,
                        help="file path or literal string; system prompts move "
                             "refusal behaviour, so bare and with-prompt are two "
                             "different measurements")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Generous on purpose. An answer truncated mid-thought "
                             "reads as non-refusal to a string matcher, and "
                             "truncation gets likelier as effort rises, which would "
                             "manufacture the very trend being measured.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-usd", type=float, default=6.0)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--probe", action="store_true",
                        help="make one call and print request, response and "
                             "environment verbatim")
    args = parser.parse_args(argv)

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("set TINKER_API_KEY (Tinker console -> API Keys)", file=sys.stderr)
        return 2

    if args.probe:
        return probe(api_key, args.model)

    system_prompt = args.system_prompt
    if system_prompt and Path(system_prompt).exists():
        system_prompt = Path(system_prompt).read_text(encoding="utf-8")

    every = load_all()
    prompts = (sorted(every, key=lambda x: x.uid) if args.per_stratum < 0
               else stratified_sample(every, args.per_stratum, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)

    jobs = [
        (p, level)
        for p in prompts
        for level in args.levels
        if (p.uid, EFFORT_LEVELS[level]) not in done
    ]
    if args.limit:
        jobs = jobs[: args.limit]

    price = PRICES.get(args.model, {"in": 1.0, "out": 4.05})
    print(describe(prompts))
    print(f"model={args.model} levels={args.levels}")
    if system_prompt:
        print(f"system prompt: {len(system_prompt)} chars")
    print(f"{len(jobs)} calls to make, {len(done)} already done -> {args.out}")

    state = {"in": 0, "out": 0, "errors": 0, "truncated": 0, "n": 0, "stop": False}
    lock = threading.Lock()

    def run_one(job):
        prompt, level = job
        if state["stop"]:
            return None
        record = {
            "uid": prompt.uid,
            "effort": EFFORT_LEVELS[level],
            "effort_name": level,
            "mode": "tinker_anthropic",
            "model": args.model,
            "source": prompt.source,
            "stratum": prompt.stratum,
            "expected": prompt.expected,
            "prompt": prompt.text,
            "has_system_prompt": bool(system_prompt),
        }
        body = build_body(
            prompt.text, level,
            model=args.model, max_tokens=args.max_tokens,
            temperature=args.temperature, system_prompt=system_prompt,
        )
        for attempt in range(args.max_retries):
            try:
                record.update(extract(call(body, api_key=api_key)))
                record.pop("error", None)
                break
            except RateLimited as exc:
                # Back off per worker; the pool keeps the others in flight.
                time.sleep(2 ** attempt + 1)
                record["error"] = f"RateLimited: {exc}"
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
                break
        return record

    with args.out.open("a", encoding="utf-8") as sink:
        def commit(record):
            if record is None:
                return
            with lock:
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                state["n"] += 1
                state["in"] += record.get("prompt_tokens") or 0
                state["out"] += record.get("completion_tokens") or 0
                if record.get("error"):
                    state["errors"] += 1
                if record.get("finish_reason") == "max_tokens":
                    state["truncated"] += 1

                cost = (state["in"] / 1e6 * price["in"]
                        + state["out"] / 1e6 * price["out"])
                n = state["n"]
                if n % 25 == 0 or n == len(jobs):
                    elapsed = max(1e-9, time.time() - t0)
                    rate = n / elapsed * 60
                    left = (len(jobs) - n) / rate if rate else 0
                    print(f"  {n}/{len(jobs)}  ~${cost:.3f}  {rate:.0f}/min  "
                          f"~{left:.0f}m left  {state['errors']} errors, "
                          f"{state['truncated']} truncated")

                # Same guards as the sequential path, just shared across workers.
                if state["errors"] >= 3 and n == state["errors"]:
                    print(f"\nFirst {n} calls all failed. Last error:\n"
                          f"  {record.get('error')}\nStopping.", file=sys.stderr)
                    state["stop"] = True
                if cost > args.max_usd:
                    print(f"\nEstimated ${cost:.2f} passed --max-usd "
                          f"{args.max_usd}. Stopping; rerun to resume.",
                          file=sys.stderr)
                    state["stop"] = True

        t0 = time.time()
        if args.concurrency <= 1:
            for job in jobs:
                if state["stop"]:
                    break
                commit(run_one(job))
                time.sleep(args.sleep)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(run_one, job): job for job in jobs}
                for future in as_completed(futures):
                    commit(future.result())

    spent_in, spent_out = state["in"], state["out"]
    errors, truncated = state["errors"], state["truncated"]
    cost = spent_in / 1e6 * price["in"] + spent_out / 1e6 * price["out"]
    print(f"done -> {args.out}")
    print(f"~${cost:.3f} spent, {errors} errors, {truncated} truncated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
