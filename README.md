# Naula

*Naula* is Finnish for a nail. This repo pins down, one by one, the claims made on 2026-10-18 about
Gate 4 of [RealTimeReadWrite](https://github.com/anttiluode/RealTimeReadWrite). That day, three AIs
read one GPU log: Claude, Sol (ChatGPT) and Gemini. Some claims were measured, some were
interpretation, and some contradicted each other. Each claim here gets:
- a script that runs in minutes on the J-lens that run already cached;
- a test and a verify/kill rule fixed in [LEDGER.md](LEDGER.md) **before any code was written**;
- a verdict: VERIFIED, KILLED or OPEN.

The constructive next step lives in Sol's WhereToWrite: memory written at the subject token, with
the model's own attention doing the routing. N2 below tests the effect that repo rests on.

| # | claim | source | verdict |
|---|---|---|---|
| N1 | G4.4's stored value was "the full stop", so the memory never held the colour | Gemini; Claude's design error | **OPEN**: no punctuation at all (the literal story is dead), but fixing the value doubles memory accuracy, 36% → 76% |
| N2 | the 56.7% effect at the name token is the model's attention routing from the *subject* | Claude, Sol, Gemini | **OPEN**: name 56.7% vs other tokens 46–48%; mostly generic leakage of a large push |
| N3 | layers 21–22 have a dead zone where a stronger push gives lower accuracy | Gemini | **VERIFIED** (shape): 100% → 33% → 100% as the push grows at layer 22; mechanism open |
| N4 | injecting the logit-lens vector lets the 0.5B model reason (66.7%), beating the in-context ceiling | Gemini | **KILLED**: the model is at chance even when told (50.5%); 66.7% shrinks to 55.2% at 8× the items |
| N5 | the J-lens reads the name at its own token in the middle layers, where the logit lens reads ~0% | the paper; seen in Gate 4 | **VERIFIED**: J 97–100% vs logit lens 0–4% at layers 0–15 |
| N6 | slow core, fast residue: the J-space part of an activation persists across tokens, the rest does not | Claude, from Sihti | **KILLED**: the residue is *more* persistent than the J-core at every layer |

## Run

Clone this next to RealTimeReadWrite. It imports that repo's `qwen/jspace.py`, so every definition
is the same as in Gate 4: model wrapper, names, templates, injection. It also loads that run's
cached J-lens.

```
E:\...\RealTimeReadWrite      (with qwen\cache\jlens_Qwen_Qwen2.5-0.5B-Instruct_128x64.pt)
E:\...\Naula

python run_all.py                       # all six, one model load (~15-30 min on an RTX 3060)
python run_all.py --only n1_full_stop   # or any subset; each nX_*.py also runs alone
python tests/test_smoke.py              # CPU, no download
```

Output goes to `results/naula_log.txt` and `results/<check>.json`, and `results/summary.json` holds
the verdict line. If RealTimeReadWrite is somewhere else, pass `--rtrw PATH`.

Fixed from Gate 4, not re-chosen:
- Qwen2.5-0.5B-Instruct;
- the 48 test and 16 dev names;
- read layer 21, write layer 16, α = 0.5.

The smoke test checks that the batched pursuit used here equals Gate 4's one-vector pursuit (max
relative difference 1e-7), and that every check runs and returns a verdict on a miniature random
Qwen2.

## Results

Run 2026-10-18 on Antti's RTX 3060: all six checks in about 13 minutes. The raw outputs are in
`results/`, and the full write-up is under each claim in [LEDGER.md](LEDGER.md).

- **Two VERIFIED:** N3 (the shape only) and N5.
- **Two KILLED:** N4 and N6.
- **Two OPEN:** N1 and N2.

In one line: the J-lens reads names natively, and writing a colour's J-lens vector works. Stored
values and the name-site effect are weaker than the day's interpretations claimed. Two of Gemini's
three strongest claims did not survive; neither did Claude's own "slow core, fast residue".
