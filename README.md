# Conversational AI portfolio

Twenty-one projects behind one hub: a grounded, evaluated, escalating support
agent, built twice — once open-source, once Google-native — with the numbers
for both.

Days 1–2 of 21 are shipped. The hub lists every remaining day and shows its
brief, so progress is visible rather than implied.

## Run it

```bash
uv sync --extra dev
uv run python -m convo.hub        # http://localhost:7860
```

`PORT` overrides the port, which is how Cloud Run starts it.

```bash
uv run pytest -q                  # 22 tests
uv run ruff check . && uv run ruff format --check .
```

## Deploy it

The hub is containerised and deploys to Cloud Run's always-free tier: 2M
requests, 180k vCPU-seconds and 360k GiB-seconds a month, scaling to zero in
between. It needs the Google Cloud CLI and a project with billing enabled.

```powershell
.\scripts\deploy.ps1
```

```bash
./scripts/deploy.sh               # SERVICE=… REGION=… to override
```

The script enables the Run, Cloud Build and Artifact Registry APIs, builds the
`Dockerfile` with Cloud Build, and prints the public URL. `--min-instances 0`
means the first request after a quiet spell takes a few seconds; the hub's
first paint is static so that reads as slow, not broken.

To check the image locally (Docker Desktop must be running):

```bash
docker build -t convo-portfolio:local .
docker run --rm -p 8081:8080 convo-portfolio:local
```

## Layout

```
src/convo/
  core/        shared primitives — no project-specific code lives here
    logging.py   one JSON object per line, so a log file is also a dataset
    tracing.py   spans per turn; day 5's trace timeline and day 12's waterfall
    costs.py     token and cost accounting, with an empty price book by design
    evals.py     EvalRun / CaseResult / Assertion — day 6 and day 18 read these
    registry.py  the 21-day plan as data, plus the contract a project implements
  hub/         the Gradio app that renders whatever is registered
  projects/    one module per day; dayNN_slug.py calls ship(day, build)
    day01_hub.py
    day02_intent_nlu/
      analysis.py  threshold + confusion maths, pure numpy, heavily tested
      data.py      CLINC150 loading and label remapping   (needs `train` extra)
      train.py     the fine-tuning CLI                    (needs `train` extra)
      results.py   loads the cached logits the UI runs on
      ui.py        the tab
      artifacts/   committed test-set logits (~3 MB) — not the weights
```

### Adding a day

Write `src/convo/projects/dayNN_slug.py` with a `build()` that lays out its
Gradio components inside the open tab, call `ship(NN, build)` at module level,
then import it in `convo/projects/__init__.py`. The hub picks it up and the
day flips from planned to shipped.

## Day 2: the out-of-scope door

The classifier is trained on CLINC150's **150 in-scope intents only**. The
dataset ships an `oos` class and most write-ups train on it as a 151st label —
that inflates the numbers, because a client's real out-of-scope traffic never
looks like a curated OOS split. Here out-of-scope is handled where it actually
has to be handled in production: at inference, by thresholding the top softmax
probability.

That turns the threshold into a product decision rather than a
hyperparameter — every point of out-of-scope recall is paid for in in-scope
accuracy — so the tab puts a slider on it and recomputes the whole analysis
live. A rejected in-scope turn is scored as **wrong, not abstained**; scoring
it any other way lets a bot look good by refusing to answer anything.

```bash
uv sync --extra train
uv run python -m convo.projects.day02_intent_nlu.train --save-model
uv run python -m convo.projects.day02_intent_nlu.train --epochs 1 --limit 600  # smoke run
```

Training writes test-set logits into `day02_intent_nlu/artifacts/` (~3 MB,
committed) and the weights into `models/` (gitignored). The UI reads only the
logits, which is why the whole threshold sweep is numpy and the hub needs no
torch. `--from-model models/day02` re-exports the artifact from saved weights
without retraining. A 150×150 confusion matrix is an unreadable smear, so the
tab shows one restricted to the worst-recall intents, with a rejection column,
plus the ranked list of confused pairs — which is the view you actually act on.

### Results

`all-MiniLM-L6-v2`, 8 epochs at 1e-4, `max_len` 24. Test set is 4,500 in-scope
turns and 1,000 out-of-scope. Validation accuracy peaked at 96.6%.

| threshold | in-scope accuracy | out-of-scope recall | false rejection |
|---|---|---|---|
| 0.00 (argmax only) | 95.8% | 0.0% | 0.0% |
| 0.50 | 95.6% | 47.5% | 1.4% |
| 0.75 | 94.6% | 69.3% | 3.2% |
| 0.90 | 93.1% | 81.4% | 5.6% |
| **0.96** (best balanced) | **90.5%** | **90.5%** | **8.6%** |
| 0.99 | 31.7% | 99.0% | 68.3% |

The useful threshold sits high — 0.96 — because a converged model is very
confident: median top probability is 0.987. That is the part worth discussing
with a client. Refusing 9% of real questions to catch 90% of nonsense is a
business tradeoff, not a tuning result, and the right point on this curve
depends on what a wrong answer costs versus what a needless "I didn't catch
that" costs.

The confusions that remain are genuinely ambiguous intent pairs rather than
noise — `transactions` → `spending_history`, `ingredients_list` → `recipe`,
`redeem_rewards` → `rewards_balance` — which is a design problem in the intent
taxonomy, not a modelling one.

### A failure worth keeping

The first run used 3 epochs at 3e-5, a sensible BERT-base recipe, and reached
74% in-scope accuracy with a **maximum softmax confidence of 0.092 across the
entire test set**. Accuracy alone understated how broken that was: with
probabilities that flat, every threshold above 0.1 rejected every single turn,
so the slider controlled nothing.

`analysis.confidence_stats()` now reports that distribution and flags
`is_degenerate` when the top probability never clears 0.5, and the tab renders
a warning above the slider when it fires. Catching this by eye once was luck;
the check makes it permanent.

## Two deliberate choices

**The price book ships empty.** `CostMeter` always counts tokens, but reports a
cost only for models you have priced yourself in `prices.toml` (see
`prices.example.toml`). `require_total_usd()` raises rather than guess. A wrong
cost figure in a portfolio is worse than no figure.

**Eval types came first.** `EvalRun.compare()` and `EvalRun.gate()` exist on
day 1 even though the first real suite is day 6, because the CI gate in day 18
has to read the same shape. Nothing about the eval format gets invented twice.

## Licence

MIT.
