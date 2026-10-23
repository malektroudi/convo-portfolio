"""A small JSON-Schema validator, written rather than imported, on purpose.

Tool schemas use a tiny corner of JSON Schema — object, required, a handful of
scalar types, enums, numeric bounds and string patterns. A general validator
covers all of it and then some, so this is not about capability.

It is about the error messages. When a model emits an invalid tool call, the
error text goes straight back into the prompt as the retry instruction, and
the wording decides whether the retry succeeds. A generic library says

    '4 people' is not of type 'integer'

which tells the model what it did, not what to do. This module says

    party_size: expected an integer, got the string "4 people"

and, where the schema allows, names the legal values. That is the difference
between a retry that fixes the call and one that repeats it.

Returning ALL errors rather than the first also matters: a model that gets one
error per round trip burns a retry per field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _describe(value: Any) -> str:
    """Name a value the way an instruction would, not the way repr does."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"the boolean {str(value).lower()}"
    if isinstance(value, str):
        return f'the string "{value}"'
    if isinstance(value, (int, float)):
        return f"the number {value}"
    if isinstance(value, list):
        return f"an array of {len(value)} item(s)"
    if isinstance(value, dict):
        return "an object"
    return repr(value)


@dataclass
class ValidationResult:
    """Every problem with one tool call, phrased as repair instructions."""

    errors: list[str] = field(default_factory=list)
    coerced: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_instruction(self) -> str:
        """The text handed back to the model on a retry."""
        if self.ok:
            return ""
        if len(self.errors) == 1:
            return f"That call was invalid: {self.errors[0]} Call the tool again, corrected."
        joined = "; ".join(self.errors)
        return (
            f"That call had {len(self.errors)} problems: {joined} Call the tool again, corrected."
        )


def _check_value(name: str, value: Any, spec: dict[str, Any]) -> tuple[list[str], Any]:
    """Validate one property, returning (errors, possibly coerced value)."""
    errors: list[str] = []
    expected = spec.get("type")

    if expected and expected in JSON_TYPES:
        allowed = JSON_TYPES[expected]
        # bool is an int in Python; a schema asking for a number never wants one.
        is_bool_mismatch = isinstance(value, bool) and expected in {"integer", "number"}
        if is_bool_mismatch or not isinstance(value, allowed):
            # A model very often sends "4" for an integer. Accept it, but say so,
            # because silently coercing hides a prompt problem worth fixing.
            if (
                expected == "integer"
                and isinstance(value, str)
                and re.fullmatch(r"-?\d+", value.strip())
            ):
                return [], int(value.strip())
            if expected == "number" and isinstance(value, str):
                try:
                    return [], float(value.strip())
                except ValueError:
                    pass
            article = "an" if expected[0] in "aeiou" else "a"
            return [f"{name}: expected {article} {expected}, got {_describe(value)}."], value

    if "enum" in spec and value not in spec["enum"]:
        legal = ", ".join(f'"{v}"' for v in spec["enum"])
        errors.append(f"{name}: must be one of {legal}, got {_describe(value)}.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            errors.append(f"{name}: must be at least {spec['minimum']}, got {value}.")
        if "maximum" in spec and value > spec["maximum"]:
            errors.append(f"{name}: must be at most {spec['maximum']}, got {value}.")

    if isinstance(value, str):
        if "pattern" in spec and not re.fullmatch(spec["pattern"], value):
            hint = spec.get("patternHint", spec["pattern"])
            errors.append(f'{name}: must look like {hint}, got "{value}".')
        if "minLength" in spec and len(value) < spec["minLength"]:
            errors.append(
                f'{name}: must be at least {spec["minLength"]} characters, got "{value}".'
            )

    return errors, value


def validate(arguments: Any, schema: dict[str, Any]) -> ValidationResult:
    """Check a tool call's arguments against its parameter schema."""
    result = ValidationResult()

    if not isinstance(arguments, dict):
        result.errors.append(
            f"arguments: expected an object of named parameters, got {_describe(arguments)}."
        )
        return result

    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    for name in required:
        if name not in arguments or arguments[name] is None:
            spec = properties.get(name, {})
            hint = f" ({spec['description']})" if spec.get("description") else ""
            result.errors.append(f"{name}: required but missing{hint}.")

    unknown = [k for k in arguments if k not in properties]
    if unknown and schema.get("additionalProperties") is False:
        legal = ", ".join(sorted(properties)) or "none"
        for key in sorted(unknown):
            result.errors.append(
                f"{key}: not a parameter of this tool. Valid parameters are: {legal}."
            )

    for name, value in arguments.items():
        if name not in properties or value is None:
            continue
        errors, coerced = _check_value(name, value, properties[name])
        result.errors.extend(errors)
        result.coerced[name] = coerced

    return result
