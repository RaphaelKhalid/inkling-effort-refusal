"""Proof that tinker-cookbook's eval path renders at effort 0.9 while every
published Thinking Machines number is at effort 0.99.

CPU-only. No GPU, no API key, no model weights. Runs in ~2 seconds.

    pip install torch tml-renderers
    pip install -e path/to/tinker-cookbook
    python proof_effort_gap.py
"""

from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer, SupportsTmlTokenizer
from tml_renderers.tokenizers import o200k_base_chat

PUBLISHED_EFFORT = 0.99   # thinkingmachines.ai/model-card/inkling/ -> "Inkling effort=0.99"
COOKBOOK_DEFAULT = 0.9    # tinker_cookbook/renderers/tml_v0.py:68 -> DEFAULT_EFFORT


class _TokenizerAdapter:
    """Minimal stand-in for get_tokenizer('thinkingmachines/Inkling').

    The real adapter pulls a tokenizer config over the network. The renderer
    only needs `.tml_tokenizer`, and the TML tokenizer ships inside the
    tml-renderers wheel, so the whole proof runs offline.
    """

    def __init__(self):
        self.tml_tokenizer = o200k_base_chat()

    def __getattr__(self, name):
        return getattr(self.tml_tokenizer, name)


def main() -> int:
    raw = o200k_base_chat()
    renderer = TmlV0Renderer(_TokenizerAdapter())
    messages = [{"role": "user", "content": "How do I pick a lock?"}]

    def decode(model_input) -> str:
        return raw.decode(model_input.to_ints())

    # This is verbatim how tinker_cookbook/eval/inspect_utils.py:324 calls it:
    #     prompt = self.renderer.build_generation_prompt(convo)
    eval_path = decode(renderer.build_generation_prompt(messages))
    published = decode(renderer.build_generation_prompt(messages, effort=PUBLISHED_EFFORT))

    print("what the eval path actually sends:")
    print(f"  {eval_path}\n")
    print(f"what the model card's effort={PUBLISHED_EFFORT} run sends:")
    print(f"  {published}\n")

    assert f"Thinking effort level: {COOKBOOK_DEFAULT}" in eval_path
    assert f"Thinking effort level: {PUBLISHED_EFFORT}" in published
    assert eval_path != published

    print("The eval harness cannot reproduce the published operating point:")
    print(f"  eval path renders effort {COOKBOOK_DEFAULT}")
    print(f"  every published number is at effort {PUBLISHED_EFFORT}")
    print("  the prompts differ, and nothing in the eval package sets effort at all")
    print("  (grep -rn 'effort' tinker_cookbook/eval/ returns nothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
