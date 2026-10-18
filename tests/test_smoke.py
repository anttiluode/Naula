"""Smoke test on a randomly initialised miniature Qwen2 (CPU, no download). Checks that the batched
pursuit equals Gate 4's one-vector pursuit, and that every check runs end to end and returns a verdict.
Numbers from a random model mean nothing.      python tests/test_smoke.py"""
import os
import re
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402

RTRW = next(p for p in (os.path.join(os.path.dirname(ROOT), "RealTimeReadWrite"),
                        os.path.join(os.path.dirname(ROOT), "realtimereadwrite")) if os.path.isdir(p))
js = common.import_jspace(RTRW)
import n1_full_stop, n2_subject_site, n3_dead_zone, n4_banana, n5_name_token, n6_slow_core  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402


class WordTok:
    def __init__(self):
        text = " ".join(js.NAME_POOL + js.COLORS + list(js.OBJECTS.values()) + js.QUESTIONS + [js.FACT, js.YESNO]
                        + js.FILLER + js.OPENINGS + n5_name_token.MENTIONS + ["Yes", "No", "n", "c", "o"])
        words = sorted(set(re.findall(r"\w+|[^\w\s]", text)))
        self.vocab = {w: i for i, w in enumerate(words)}
        self.inv = {i: w for w, i in self.vocab.items()}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[w] for w in re.findall(r"\w+|[^\w\s]", s)]

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids)


fails = 0
def check(name, ok, info=""):
    global fails
    print(f"{'ok  ' if ok else 'FAIL'} {name} {info}", flush=True)
    fails += 0 if ok else 1


torch.manual_seed(0)
tok = WordTok()
cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True)
lm = Qwen2ForCausalLM(cfg).eval()
fz = js.Frozen(lm, tok, "cpu")
Js = js.jlens(fz, js.corpus(fz, 2, 8, seed=0), rows_per_pass=16, log=lambda s: None)

# batched pursuit == Gate 4's single-vector pursuit
g = torch.Generator().manual_seed(2)
D = torch.nn.functional.normalize(torch.randn(300, 32, generator=g), dim=-1)
H = torch.randn(7, 32, generator=g)
comp, sel, coef = common.batched_component(D, H, k=8)
single = types.SimpleNamespace(D=D, k=8)
errs = []
for i in range(7):
    c1, s1, _ = js.JSpace.component(single, H[i])
    errs.append(((comp[i] - c1).norm() / c1.norm().clamp_min(1e-8)).item())
check("batched pursuit matches Gate 4's pursuit", max(errs) < 1e-3, f"max rel diff {max(errs):.2e}")
check("batched pursuit coefficients nonnegative", bool((coef >= 0).all()))

c = common.Ctx(js, fz, Js, lr=2, lw=1, alpha=0.5, log=print, small=True)
for name, mod in (("n1", n1_full_stop), ("n2", n2_subject_site), ("n3", n3_dead_zone), ("n4", n4_banana),
                  ("n5", n5_name_token), ("n6", n6_slow_core)):
    res = mod.run(c)
    check(f"{name} runs and returns a verdict", isinstance(res.get("verdict"), str), res["verdict"])
sys.exit(1 if fails else 0)
