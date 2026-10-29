"""Find a Gemini model this key can actually call, and cache the answer.

Hardcoding a model name looked obviously right and is obviously wrong here.
Verifying the key on the day it was created found that a new free-tier key
lists 46 models and cannot call several of the ones it lists:

    gemini-2.5-flash        404  "no longer available to new users"
    gemini-2.5-flash-lite   404  same
    gemini-3.8-flash        503  temporarily unavailable
    gemini-3-flash-preview  answers

A name that works today is a name that 404s for whoever runs this next, so
the model is resolved by asking, once, and the answer is written to the
artifact so a reader knows which model produced the numbers.

The key itself is read from `.env`, which is gitignored. Nothing here prints
it, and `redact` exists because an API error can carry the request URL and
the URL carries the key.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from functools import lru_cache

from convo.core import get_logger

ENV_FILE = pathlib.Path(__file__).resolve().parents[4] / ".env"
PREFERENCE = (
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
)
"""Tried in order, then any other flash model the key lists.

Lite first, for a reason discovered the expensive way. The binding limit on a
free key is not tokens and not requests per minute, it is
`GenerateRequestsPerDayPerProjectPerModel`, and it is **not the same number
for every model**. Measured off the 429 bodies, not read from documentation:

    gemini-3.5-flash-lite     500 requests per day
    gemini-3.5-flash           20 requests per day

Twenty-five times apart, for models one word apart in name. Day 19 found out
what that means in practice: a cross-model run on `flash` reached **one**
scenario of 32 before the wall, so the comparison it was supposed to make was
not merely imprecise, it was unreachable — the five conversations the day is
about sit at suite positions 12 to 20.

So lite is not "the cheap option" on this tier. It is the only model with
enough daily headroom to run a 32-scenario suite at all, and any experiment
that varies the model is off the free tier by construction.

It is also the right model for the job independently: a booking bot does
shallow reasoning under a latency budget, which is what lite is for. The
quota only forced a choice that was defensible anyway."""

log = get_logger("day16.model")


@dataclass(frozen=True)
class Resolved:
    model: str
    listed: int
    rejected: tuple[str, ...]
    """Models that were listed and then refused the call, with the reason."""


def load_key() -> str:
    """Read GOOGLE_API_KEY from the environment, falling back to `.env`."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == "GOOGLE_API_KEY":
            return value.strip().strip("\"'")
    return ""


def redact(text: object, key: str = "") -> str:
    """Strip the key out of anything on its way to a log or a terminal."""
    out = str(text)
    key = key or load_key()
    if key:
        out = out.replace(key, "<redacted>")
        if len(key) > 12:
            out = out.replace(key[:12], "<redacted>")
    return out


def ensure_env() -> str:
    """Put the key in the environment, which is where ADK looks for it.

    ADK constructs its own `genai` client deep inside the runner and reads
    `GOOGLE_API_KEY` from the process environment; it never sees a key passed
    to code up here. Reading `.env` into a variable therefore authenticates
    nothing, which is exactly how this failed the first time.

    The Vertex switches are cleared at the same time: if either is set the
    SDK routes to Vertex, which needs the billing account this project does
    not have.
    """
    key = load_key()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
    for vertex in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        os.environ.pop(vertex, None)
    return key


def available() -> bool:
    return bool(load_key())


class ModelUnavailable(RuntimeError):
    """A specific model was asked for by name and would not serve.

    Separate from `resolve()`'s fallback on purpose. Falling back is right when
    the question is "run this bot somewhere"; it is wrong whenever a figure is
    being compared against an earlier run, because the model is the controlled
    variable and a silent substitution turns a topology comparison into a model
    comparison that still looks like the first one.

    Day 19 found this the cheap way, by probing before running: `flash-lite`'s
    daily quota was gone, `resolve()` had quietly moved to `flash`, and a full
    32-scenario run would have produced an artifact that compared cleanly
    against day 16 and meant nothing.
    """


def require(name: str) -> Resolved:
    """Resolve exactly `name`, or raise rather than substituting.

    One call, so a wrong answer costs a token or two instead of a run.
    """
    from google import genai
    from google.genai import types

    key = ensure_env()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set; put it in .env")

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    try:
        client.models.generate_content(
            model=name,
            contents="ok",
            config=types.GenerateContentConfig(max_output_tokens=4),
        )
    except Exception as exc:  # noqa: BLE001 - the message may carry the key
        detail = redact(exc, key)
        daily = "perday" in detail.lower().replace(" ", "").replace("_", "")
        raise ModelUnavailable(
            f"{name} would not serve"
            + (" (daily quota exhausted; it resets after 24h)" if daily else "")
            + f": {detail[:200]}"
        ) from exc

    log.info("model pinned", extra={"model": name})
    return Resolved(name, 0, ())


@lru_cache(maxsize=1)
def resolve() -> Resolved:
    """The first preferred model that answers a one-token request.

    **Falls back on failure, including on an exhausted daily quota.** That is
    what makes it unsuitable for a run whose figures will be compared against
    an earlier one — use `require()` there.

    Listing is not enough — the listing includes models the key may not call —
    so each candidate is actually invoked. That costs a handful of tokens and
    removes a whole class of failure that otherwise appears halfway through a
    suite run.
    """
    from google import genai
    from google.genai import types

    key = ensure_env()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set; put it in .env")

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    listed = sorted(
        {
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        }
    )

    def rank(name: str) -> tuple[int, str]:
        for index, prefix in enumerate(PREFERENCE):
            if name.startswith(prefix):
                return index, name
        return len(PREFERENCE), name

    candidates = [
        name
        for name in sorted(listed, key=rank)
        if "flash" in name
        and not any(x in name for x in ("image", "tts", "audio", "native", "omni"))
    ]

    rejected: list[str] = []
    for name in candidates[:8]:
        try:
            client.models.generate_content(
                model=name,
                contents="ok",
                config=types.GenerateContentConfig(max_output_tokens=4),
            )
        except Exception as exc:  # noqa: BLE001 - the message may carry the key
            rejected.append(f"{name}: {type(exc).__name__}")
            continue
        log.info("model resolved", extra={"model": name, "listed": len(listed)})
        return Resolved(name, len(listed), tuple(rejected))

    raise RuntimeError(
        "no listed flash model accepted a request. Tried: " + "; ".join(rejected[:8])
    )
