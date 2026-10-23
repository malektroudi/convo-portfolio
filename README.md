# Conversational AI portfolio

Twenty-one projects behind one hub: a grounded, evaluated, escalating support
agent, built twice — once open-source, once Google-native — with the numbers
for both.

Day 1 of 21 is shipped. The hub lists every remaining day and shows its brief,
so progress is visible rather than implied.

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
```

### Adding a day

Write `src/convo/projects/dayNN_slug.py` with a `build()` that lays out its
Gradio components inside the open tab, call `ship(NN, build)` at module level,
then import it in `convo/projects/__init__.py`. The hub picks it up and the
day flips from planned to shipped.

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
