"""Token and cost accounting.

Deliberately ships with an EMPTY price book. Model prices change often and a
wrong number in a portfolio is worse than no number, so you register prices
yourself from each provider's own pricing page (see prices.example.toml) and
this module records what was actually spent. Tokens are always counted; cost
is reported only for models you have priced.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class UnpricedModel(KeyError):
    """Raised only when a cost is demanded for a model with no registered price."""


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens, the unit every provider publishes."""

    input_per_mtok: float
    output_per_mtok: float

    def usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass
class Call:
    model: str
    input_tokens: int
    output_tokens: int
    usd: float | None
    label: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostMeter:
    """Accumulates model usage across a run, a conversation, or an eval suite."""

    prices: dict[str, Price] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> CostMeter:
        """Load a price book. Each entry sets input_per_mtok and output_per_mtok."""
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        prices = {
            name: Price(float(row["input_per_mtok"]), float(row["output_per_mtok"]))
            for name, row in data.get("models", {}).items()
        }
        return cls(prices=prices)

    def register_price(self, model: str, input_per_mtok: float, output_per_mtok: float) -> None:
        self.prices[model] = Price(input_per_mtok, output_per_mtok)

    def record(self, model: str, input_tokens: int, output_tokens: int, label: str = "") -> Call:
        price = self.prices.get(model)
        call = Call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=price.usd(input_tokens, output_tokens) if price else None,
            label=label,
        )
        self.calls.append(call)
        return call

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_usd(self) -> float:
        """Cost of the priced calls only. Check `unpriced` before quoting this."""
        return round(sum(c.usd for c in self.calls if c.usd is not None), 6)

    @property
    def unpriced(self) -> set[str]:
        return {c.model for c in self.calls if c.usd is None}

    def require_total_usd(self) -> float:
        """Total cost, or raise if any call was unpriced. Use this in reports."""
        if self.unpriced:
            raise UnpricedModel(f"no price registered for: {sorted(self.unpriced)}")
        return self.total_usd

    def by_model(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for call in self.calls:
            row = out.setdefault(
                call.model,
                {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "usd": 0.0},
            )
            row["calls"] += 1
            row["input_tokens"] += call.input_tokens
            row["output_tokens"] += call.output_tokens
            row["usd"] = round(row["usd"] + (call.usd or 0.0), 6)
        return out
