# Naula — the ledger

*Naula* is Finnish for a nail. This repo nails down claims made on 2026-10-18 about Gate 4 of
[RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite). The claims came from three
AIs: Claude, Sol (ChatGPT) and Gemini. Each claim gets one script that runs in minutes on the J-lens
already cached by that run, and one line in this ledger: **VERIFIED**, **KILLED** or **OPEN**.

This file was written and committed **before any of the code**. The tests and verdict rules below
are fixed; results are added under each claim without changing them.

Common setup, inherited from the Gate 4 run so nothing is re-chosen:
- model `Qwen/Qwen2.5-0.5B-Instruct`, fp32, frozen;
- J-lens: 128 prompts × 64 tokens, the cached file from RealTimeReadWrite;
- the 48 test names and 16 dev names, split exactly as in Gate 4;
- the registered choices: read layer ℓ_r = 21, write layer ℓ_w = 16, α = 0.5
  (inject α‖h‖ along a unit direction).

J-space component = 16-atom nonnegative pursuit over unit J-lens vectors of the whole vocabulary,
batched on the GPU. The smoke test checks it against Gate 4's one-vector version.

---

## N1 — "The stored value was the full stop" (Gemini; Claude's design error)

**Claim.** G4.4's memory stored the J-space content at the colour token. In a causal model that
position mostly predicts the next token ("."), so the memory never held "yellow". That would be why
even the supplied-key arm reached only 35.6%.

**Test.**
- (a) For every test name × colour, take the stored value as Gate 4 defined it (J-space component of
  h_16 at the colour token). Read it through the J-lens and report:
  - the fraction whose top-1 vocabulary token is punctuation;
  - the fraction whose top colour (among the four) is the stored colour.
- (b) Rerun G4.4's supplied-key arm with the identical protocol (N = 8, 20 episodes, one overwrite
  each, filler, latest-colour scoring) in two versions:
  - stored value = Gate 4's captured value (should reproduce about 35.6%);
  - stored value = the colour's own unit J-lens vector at layer 16.
  Also report the mean off-diagonal cosine of the supplied keys.

**Verdict.**
- VERIFIED if the J-lens-value version reaches ≥ 80% while the captured value stays ≤ 50%.
- KILLED if the J-lens-value version is < 50%: then a correct value and a correct key still fail,
  and something else breaks the read.
- OPEN otherwise.

**Result (2026-10-18, Antti's RTX 3060): OPEN by the registered rule. The literal "full stop" story is KILLED.**
- (a) Read through the J-lens, the stored value is **never** punctuation as its top token (0 of 192).
  The stored colour wins among the four 83.9% of the time. But the top tokens are a few generic
  variants: "-green" 93 times, "\tcolor" 47, ".Blue" 28, "-blue" 15. The value is a colour-ish
  smear pulled toward one attractor, not "yellow".
- (b) With the same episodes and keys, the captured value scores **36.2%** (Gate 4 reported 35.6%,
  so it reproduces) and the colour's own J-lens vector scores **76.2%**. The memory output itself
  carries the right colour 63.1% vs 83.1% of the time.
  - That is 4 points short of the registered 80%, hence OPEN.
  - The value accounts for 40 points. The remaining gap is the keys: the supplied name keys overlap
    with a mean cosine of 0.32, so 8 stored facts bleed into each other.

## N2 — "The 56.7% at the name token is the model's attention routing from the subject"

**Claim.** A colour planted on the name token at layer 16 reaches the answer because the question
position attends to the subject.

**Test.**
- Re-select the site-S gain on dev names: α ∈ {0.5, 1, 2, 4, 8}, best four-colour accuracy, ties to
  the smaller α. Gate 4's choice was not saved; the run was stopped before writing its JSON.
- With that α at layer 16, on test names (48 names × 3 questions × 3 non-own targets), inject at:
  - the name token;
  - control: the token " color" (present in all three questions);
  - control: the first token of the question.

**Verdict.**
- VERIFIED (subject-specific) if name − max(controls) ≥ 20 points.
- KILLED if max(controls) ≥ name − 5 points: then any earlier token works, which is attention to
  anything, not to the subject.
- OPEN otherwise.

**Result: OPEN, and mostly generic leakage.**
- Dev selected α = 8, eight times the residual norm.
- Test, 432 trials per site: name token **56.7%**, " color" token **48.4%**, first token **45.8%**
  (reference: question end 100%).
- The name beats the best control by 8.3 points: above the kill line (5), below the verify line (20).
- Most of the 56.7% is any large push on any earlier token leaking into the answer. A small
  subject-specific part (about 8 points, roughly 2.4 standard errors) remains.
- For WhereToWrite, the name site is not yet shown to be special.

## N3 — "Layers 21–22 have a dead zone where a stronger push gives lower accuracy" (Gemini)

**Test.** Layers 20, 21, 22; α ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8}; test names, same trials
as G4.2. For each failure, record where the answer went: another colour, or a non-colour top token.

**Verdict.**
- VERIFIED (non-monotonic) if at layer 21 or 22 some α_lo < α_hi ≤ 2 has
  accuracy(α_lo) − accuracy(α_hi) ≥ 30 points.
- KILLED otherwise.
- "Homeostatic suppression" as a *mechanism* stays OPEN either way. This test only establishes the
  shape.

**Result: VERIFIED (the shape).** The largest drop is 66.9 points, at layer 22, from α 0.25 (100%)
to α 2 (33.1%).
- Layer 22: 100% at α = 0.25, then 33–42% for α from 0.5 to 2, then 91–100% for α ≥ 3.
- Layer 21: 100% at α ≤ 0.5, falling to 50–64% at α 1–2, back to 100% at α ≥ 3.
- Layer 20 keeps four-colour accuracy at 98.6–100% everywhere. But the share of trials whose top-1
  token is a colour dips to 40% at α = 2.
- Where the failures go, at layer 22 for α 0.5–2: four-colour accuracy, strict accuracy and "top-1
  is a colour" are all 33.1%. So whenever any colour comes out on top it is the target. In the other
  two thirds, the top token is not one of the four colours at all.
- Mechanism ("homeostatic suppression"): OPEN, as registered. The data fits "moderate pushes at
  layers 21–22 produce non-colour tokens", which matches the "-green" / ".Blue" variants seen in N1a.

## N4 — "Injecting the logit-lens vector lets the 0.5B model reason (66.7%), beating the in-context ceiling (50%)" (Gemini)

**Test.**
- Items: every test name × every colour × {one matching object, one non-matching object}, giving
  384 balanced items. "Is {n} the same color as {obj}? Answer:", scored as logit(" Yes") > logit(" No").
- Arms: ceiling (fact in context), none, J (layer 16, α 0.5), logit lens (layer 16, α 0.5).
- Balanced accuracy, with 95% bootstrap CIs over names (2,000 resamples).

**Verdict.**
- If the ceiling's CI does not lie above 60%, the model cannot do the task when told. Every
  reasoning claim is then VOID, and N4 is KILLED as stated.
- Otherwise: VERIFIED if logit − none ≥ 10 points with the CI of the difference excluding 0;
  KILLED if that CI includes 0.

**Result: KILLED (VOID).**
- Ceiling (fact in context) **50.5%** [49.5, 51.6]: Qwen-0.5B cannot do the task even when told.
- None 50.3%, J 46.4% [44.3, 48.4], logit 55.2% [53.1, 57.3].
- Logit − none = 4.9 points [2.6, 7.0]: statistically present but small. With the ceiling at chance,
  it is a shift in yes/no bias, not reasoning.
- Gate 4's 66.7% came from 48 items; with 384 items it shrinks to 55.2%.

## N5 — "The J-lens reads the name at its own token in the middle layers, where the logit lens reads ~0%" (replication of the paper's central claim)

**Test.** Test names in 8 neutral mention sentences (not the Gate 4 questions). At the name token,
classify the name by cosine with the 48 candidates' unit J-lens vectors and, separately, logit-lens
vectors, at every layer.

**Verdict.**
- VERIFIED if J ≥ 90% and logit lens ≤ 20% at every layer from 6 to 15.
- KILLED if the logit lens ≥ 50% at most of those layers (then the J-lens adds nothing there).
- OPEN otherwise.

**Result: VERIFIED.**
- At the name token, in 8 neutral sentences (384 cases): the J-lens identifies the name 97–100% at
  every layer 0–15; the logit lens scores 0–4%.
- Identity fades after layer 15 (89, 82, 83, 80%), then collapses at layers 20–23 (28, 17, 7, 10%).
- This replicates the paper's central claim on a small open model: the J-lens recovers content
  where the logit lens sees nothing.

## N6 — "Slow core, fast residue": the J-space component of an activation persists across tokens; the rest of the same activation does not

**Background.** The paper (Fig. 28c) already shows that the J-lens top-1 token persists across
positions in the middle layers. "The workspace is slow" is theirs. New here: whether the
*remainder* of the same activation, h − J(h), is fast. That would make "slow core, fast residue"
(the Sihti split, along the token axis) a measured decomposition.

**Test.**
- 32 fresh prompts × 64 tokens sampled from the model (seed 1, not the lens corpus); position 0
  dropped (attention sink).
- Layers {4, 8, 12, 16, 20, 23}. At each, split h_t = c_t + r_t, with c = J-space component.
- Control: a size-matched sparse component from a random dictionary (same shape as the vocabulary,
  random unit atoms, same pursuit).
- Persistence ρ(Δ) = mean cos(x_t − μ, x_{t+Δ} − μ) for Δ = 1, 4, 16. μ is the global mean of that
  component at that layer. Each value is reported minus a position-shuffled null (random t' in the
  same prompt).

**Verdict.**
- VERIFIED if, at every workspace layer among {12, 16, 20}, the excess persistence at Δ = 1 of c
  exceeds that of r by ≥ 0.10 AND exceeds the random-dictionary control by ≥ 0.10.
- KILLED if r is at least as persistent as c at those layers.
- OPEN otherwise.

**Result: KILLED.** At every layer the *residue* is more persistent from one token to the next than
the J-space core:

| layer | 4 | 8 | 12 | 16 | 20 | 23 |
|---|---|---|---|---|---|---|
| J-core | +0.119 | +0.163 | +0.186 | +0.203 | +0.182 | +0.119 |
| residue | +0.183 | +0.261 | +0.284 | +0.263 | +0.239 | +0.203 |
| random-dictionary core | +0.045 | +0.067 | +0.075 | +0.072 | +0.067 | +0.073 |

Values are excess persistence at Δ = 1. The J-core holds 10–16% of the activation energy in layers
4–20, consistent with the paper's "small fraction".

Observation, not a registered claim:
- The J-core's same-prompt null is higher than the residue's (0.158 vs 0.114 at layer 16). Its
  content is more shared across the whole prompt.
- The residue is smoother from one token to the next.
- At Δ = 16 both are back at the null.

So the workspace looks *prompt-wide* rather than *locally slow*. "Slow core, fast residue" is wrong
as stated.

---

Results go below each claim as they come in, with the date and the machine they ran on.
