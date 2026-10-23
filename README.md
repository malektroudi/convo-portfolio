# Conversational AI portfolio

Twenty-one projects behind one hub: a grounded, evaluated, escalating support
agent, built twice — once open-source, once Google-native — with the numbers
for both.

Days 1–5 of 21 are shipped. The hub lists every remaining day and shows its
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
    logging.py       one JSON object per line, so a log file is also a dataset
    tracing.py       spans per turn; day 5's timeline and day 12's waterfall
    costs.py         token and cost accounting, empty price book by design
    evals.py         EvalRun / CaseResult / Assertion — days 6 and 18 read these
    classification.py  threshold + confusion maths over any score array whose
                       row max is a confidence; day 2 feeds it softmax, day 3
                       cosine similarity, so both are scored by one code path
    registry.py      the 21-day plan as data, plus the project contract
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
    day03_embed_router/
      router.py    nearest-neighbour intent routing, pure numpy, no training
      analysis.py  segment-wise scoring: known / new / out-of-scope
      embed.py     base-encoder sentence embeddings   (needs `train` extra)
      benchmark.py the held-out-intent experiment     (needs `train` extra)
      results.py   loads the cached comparison
      ui.py        the tab
      artifacts/   both approaches' score matrices (~6 MB)
    day04_dialog_engine/
      extract.py   deterministic parsers — dates, times, counts, names
      flow.py      the flow as data: pages, slots, validators, digressions
      engine.py    the state machine; one turn in, one TurnResult out
      ui.py        chat beside a live state inspector
    day05_tool_agent/
      schema.py    a small JSON-Schema validator, written for its error text
      grounding.py the provenance guard — day 4's parsers as an oracle
      tools.py     book_table / get_policy, same rules as day 4's validators
      agent.py     the loop: parse, validate, ground, retry, stop
      backends.py  Scripted (tests) and LocalTransformers (`train` extra)
      scenarios.py the shared suite both approaches are scored on
      benchmark.py runs both, twice                     (needs `train` extra)
      ui.py        the three-way comparison
      artifacts/   the cached comparison (~200 KB)
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

## Day 3: when fine-tuning is worth it, and when it isn't

Ten of the 150 intents were **withheld from the classifier entirely**, so it
trained on 140 classes and its output layer has no slot for the rest. The
router indexed those same 140 in full plus **five examples each** of the ten
new ones, using the base encoder with no fine-tuning anywhere. Both were then
scored on the untouched test set.

Simulating this would have been cheating. Asking a 150-class model about a
151st intent is not a hard question for it, it is an impossible one, and the
only honest way to show that is to make the new intents genuinely absent.

```bash
uv sync --extra train
uv run python -m convo.projects.day03_embed_router.benchmark
```

### Results

Each approach at its own best segment-balanced threshold (router 0.59 cosine,
classifier 0.95 softmax — they are not the same scale, which is why the tab has
two sliders):

| segment | turns | embedding router | fine-tuned classifier |
|---|---|---|---|
| Known intents | 4,200 | 84.1% | **91.2%** |
| **New intents** | 300 | **52.3%** | **0.0%** |
| Out-of-scope refused | 1,000 | 85.7% | **89.1%** |

Without any rejection (pure argmax), known-intent accuracy is 88.5% for the
router against 96.1% for the classifier; on the new intents it is 60.7% against
a structural zero.

So fine-tuning buys about **7 points** on intents you have data for. It costs
you the ability to answer a new one at all until you retrain.

| operational cost | router | classifier |
|---|---|---|
| Add one new intent | **7.6 ms** | 43.6 min (full retrain) |
| Index / model build | 31 ms | 43.6 min |
| Per-query latency | slower — same encoder pass, then an index search | faster |

Adding an intent is roughly **345,000× faster** on the router. Note the latency
row though: the router is *slower* per query, because it runs the same encoder
forward pass and then searches an index. Its win is iteration speed, not
inference speed — a distinction the usual sales pitch skips.

### The finding that actually matters

The 52.3% average on new intents is close to meaningless, because the spread
across those ten intents is nearly the whole range:

| new intent | router recall from 5 examples |
|---|---|
| `order_checks` | 96.7% |
| `next_song` | 90.0% |
| `change_speed` | 86.7% |
| `oil_change_when` | 83.3% |
| `smart_home` | 53.3% |
| `exchange_rate` | 43.3% |
| `user_name` | 36.7% |
| `thank_you` | 30.0% |
| `make_call` | 3.3% |
| `definition` | 0.0% |

Five examples are plenty for `order_checks`, whose vocabulary is distinctive,
and useless for `definition`, which is phrased like half the other intents in
the dataset ("what does X mean"). The distribution is roughly bimodal, not
centred on the average.

The practical rule that falls out: before promising a client that a new intent
needs only a handful of examples, check what its nearest neighbours in the
existing taxonomy are. Few-shot routing works when an intent occupies its own
region of embedding space and fails when it overlaps an existing one — and you
can measure that before collecting any data.

## Day 4: the dialogue engine, with no model in it

A restaurant booking as an explicit state machine — pages, a form of typed
parameters, per-slot validators, a confirmation step, routes between them.
Intent and entity handling are patterns and parsers, not a model. This is the
same shape Conversational Agents (formerly Dialogflow CX) gives you, written
out so the mechanism is visible rather than drawn in a console; **day 19
rebuilds this exact flow in the managed product and compares the two.**

Nothing to train and no artifact to build — `uv run python -m convo.hub` and
open Week 1 → day 04. Four scripted conversations replay from buttons.

### The two cases that matter

Everything else in a booking bot is bookkeeping. These two are where demos
fall over:

**Repair.** At the confirmation, "no, make it 8pm" has to change the time and
re-confirm — not read the "no" as a rejection and dump the user into a menu.
The engine checks for a correction *before* a plain yes/no, so a denial
carrying a value is a repair and a bare denial is an amend:

```
bot:  So that's 4 people on Friday 13 March at 7:30pm under Sarah. Shall I book it?
user: no, make it 8pm
bot:  Changed the time. So that's 4 people on Friday 13 March at 8pm under Sarah. Shall I book it?
```

**Digression.** An off-task question mid-form has to be answered and then
resume at the *same* slot, with the form intact:

```
user: a table for 4
bot:  What day would you like?
user: do you have parking?
bot:  There's paid street parking on Dundas, and a garage one block east.
      What day would you like?
```

Digressions are checked before form filling, so "do you have parking?" is
never mined for a date, and they don't count against the failed-prompt counter
that escalates to a human.

### Decisions worth defending

**Ambiguous input is refused, not guessed.** `3/4` is April 3rd to most of the
world and March 4th in the US. The parser returns nothing and the bot asks,
which costs one turn; guessing books the wrong night. ISO dates are accepted
because they are unambiguous.

**Names need a carrier phrase.** "my name is sarah" and "under okonkwo" parse;
a bare word does not. Free-form name guessing writes *Friday* on the
reservation.

**A rejected value keeps the old one.** An invalid correction at the
confirmation ("no, make it 3am") is refused and the existing time survives, so
a bad repair cannot empty a filled form.

**Three failed prompts escalate.** Not a retry loop — the counter resets on
any progress, and digressions don't count. Day 13 builds the real handoff on
top of this.

### Observability is the deliverable

Every turn returns a `TurnResult` recording what was extracted, which slots
filled or were overwritten, which were rejected and why, the page move, the
transition reason and the turn time. That record is what the live inspector
renders, and it is what day 6's eval harness will assert against — the engine
is observable by construction rather than by adding logging afterwards.

The whole thing is deterministic: the clock is injected rather than read, so
the same transcript always produces the same slots and the same replies. A
test asserts exactly that, because without it an eval suite over this engine
would be flaky by design.

## Day 5: the same bot, tool-calling

Day 4's booking flow rebuilt as an LLM with tools, then both run over the
**same 12 scripted conversations** and scored by the **same assertions**. The
tools enforce the same business rules as day 4's validators, so neither side
can look better by being allowed more. Model is `Qwen/Qwen2.5-0.5B-Instruct`,
greedy, on CPU — 1.5B took over ten minutes for a single generation here, and
0.5B is what a free tier realistically gives you anyway.

```bash
uv sync --extra train
uv run python -m convo.projects.day05_tool_agent.benchmark
```

### What the first run found

Not what I expected. Across 20 model calls there was **one** malformed call and
**zero** unparseable ones — parsing and schema validation were barely doing any
work. What the model did instead was call `book_table` on the *opening*
utterance with a complete, well-formed, entirely invented booking. Given "a
table for 4" it booked 4 people for today at 7:30pm under **"John Doe"**. Asked
for a table for 40, it booked 12 under **"Table 12"**.

Ten of twelve conversations ended in a hallucinated booking, and every one of
them passed the validator, because **inventing plausible data is not a schema
violation**.

That is the structural difference between the two architectures, and it is
sharper than any accuracy number:

> Day 4 cannot hallucinate a slot — a slot is only ever filled by an extractor
> reading the guest's own words. Day 5 can, because the model writes the
> arguments itself. Schema validation checks **shape**, not **provenance**.

### The guard, and what it actually bought

So `grounding.py` adds a third check: before a booking executes, every argument
must be traceable to something the guest said — using **day 4's own parsers as
the oracle**. The deterministic component vouches for the generative one, which
is the honest answer to "deterministic or generative?".

| | state machine | agent, schema only | agent, + provenance |
|---|---|---|---|
| Pass rate | 100%\* | 8.3% | 8.3% |
| **False confirmations** | **0** | **10** | **0** |
| Failed to book when it should | 0 | 1 | 11 |
| Model calls | 0 | 20 | 135 |
| Wall time | <1s | 4 min | 34 min |

\* written from day 4's semantics, so partly by construction.

**The pass rate did not move.** On that number the guard bought nothing. But
false confirmations went from 10 to 0: it converted **silent wrong bookings
into visible failures**. A bot that says it cannot help is categorically
different from one that sends someone to a restaurant not expecting them, and a
single aggregate score treats those two outcomes identically.

That is the lesson, and it is why day 6 scores several dimensions instead of
one. If the only number you track is pass rate, you cannot see a safety fix at
all — you would conclude the guard was a waste and delete it.

It is not free either: 20 model calls became 135 and four minutes became
thirty-four, because a 0.5B model that cannot ground its answer retries until
the ceiling stops it. That cost is the price of refusing to lie, and it is a
cost a bigger model would mostly avoid.

### Caveats worth stating

The model is tiny, and a frontier model would hallucinate far less — the
absolute numbers here are about 0.5B, not about tool-calling agents in general.
What generalises is the *shape* of the finding: schema validation does not
constrain semantics, provenance checking does, and pass rate hides the
difference.

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
