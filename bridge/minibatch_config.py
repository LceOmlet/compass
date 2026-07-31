from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def minibatch_config_kwargs(
    values: Mapping[str, Any],
    *,
    namespace: str,
) -> dict[str, int | None]:
    """Validate one legacy size or an explicit proposal/admission pair."""

    legacy_field = "reflection_minibatch_size"
    split_fields = ("proposal_minibatch_size", "admission_minibatch_size")
    legacy = legacy_field in values
    present_split = {name for name in split_fields if name in values}
    if legacy and present_split:
        raise ValueError(
            f"{namespace}.{legacy_field} cannot be combined with "
            f"{split_fields[0]} or {split_fields[1]}"
        )
    if not legacy and present_split != set(split_fields):
        raise ValueError(
            f"{namespace}.{split_fields[0]} and {split_fields[1]} must be "
            f"provided together; otherwise {legacy_field} is required"
        )
    selected = (legacy_field,) if legacy else split_fields
    for name in selected:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"{namespace}.{name} must be a positive integer")
    if legacy:
        return {
            legacy_field: values[legacy_field],
            split_fields[0]: None,
            split_fields[1]: None,
        }
    return {
        legacy_field: None,
        split_fields[0]: values[split_fields[0]],
        split_fields[1]: values[split_fields[1]],
    }


__all__ = ["minibatch_config_kwargs"]
