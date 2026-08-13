from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tinycable.core.field import Field


Filter = Mapping[str, Callable[[Any], bool]]
_FILTER_KEYS = {"field", "role", "shape", "slot", "sites", "value"}


def _python(value: Any) -> Any:
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value.tolist()


def _value_key(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def _compact_indices(values: list[int]) -> str:
    if not values:
        return "[]"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = value
        previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return f"[{', '.join(ranges)}]"


def _text(value: Any, *, indices: bool = False) -> str:
    if value is None:
        return "-"
    if indices and isinstance(value, list):
        return _compact_indices(value)
    return value if isinstance(value, str) else repr(value)


def _compress_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(row["slot"] is None for row in rows):
        return rows

    slot_sites = {row["slot"]: row["sites"] for row in rows}
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = _value_key(row["value"])
        if key not in grouped:
            grouped[key] = {"slot": [], "value": row["value"]}
        grouped[key]["slot"].append(row["slot"])

    for row in grouped.values():
        sites = [slot_sites[slot] for slot in row["slot"]]
        row["sites"] = (
            None if any(site is None for site in sites) else sorted(sum(sites, []))
        )
    return list(grouped.values())


def field_dict(
    field: "Field",
    *,
    role: bool = False,
    shape: bool = False,
    slot: bool = True,
    sites: bool = True,
    filter: Filter | None = None,
    compress: bool = False,
) -> dict[str, Any]:
    """Return a filtered per-slot representation of one Field."""
    predicates = {} if filter is None else dict(filter)
    unknown = set(predicates) - _FILTER_KEYS
    if unknown:
        raise ValueError(f"unknown Field filters: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    if field.index is None:
        rows.append({"slot": 0, "sites": None, "value": _python(field.slots[0])})
    elif field.n_slots:
        for slot_id in range(field.n_slots):
            rows.append(
                {
                    "slot": slot_id,
                    "sites": field.index[field.groups == slot_id].tolist(),
                    "value": _python(field.slots[slot_id]),
                }
            )
    else:
        rows.append({"slot": None, "sites": [], "value": []})

    data: dict[str, Any] = {
        "field": field.ref,
        "role": "state" if field.dynamic else "param",
        "shape": field.value_shape,
        "rows": rows,
    }
    for key in ("field", "role", "shape"):
        if key in predicates and not predicates[key](data[key]):
            return {}
    rows = [
        row
        for row in rows
        if all(
            predicate(row[key]) for key, predicate in predicates.items() if key in row
        )
    ]
    if not rows:
        return {}
    if compress:
        rows = _compress_rows(rows)
    data["rows"] = rows
    if not role:
        data.pop("role")
    if not shape:
        data.pop("shape")
    if not slot:
        for row in rows:
            row.pop("slot", None)
    if not sites:
        for row in rows:
            row.pop("sites", None)
    return data


def fields_dict_to_str(fields: Mapping[str, Mapping[str, Any]]) -> str:
    """Render Field dictionaries as one aligned per-slot table."""
    if not fields:
        return ""

    first = next(iter(fields.values()))
    headers = ["field"]
    headers.extend(key for key in ("role", "shape") if key in first)
    headers.extend(key for key in ("slot", "sites", "value") if key in first["rows"][0])
    rows: list[list[str]] = [headers]
    for data in fields.values():
        for row in data["rows"]:
            rows.append(
                [_text(data["field"])]
                + [_text(data[key]) for key in ("role", "shape") if key in data]
                + [
                    _text(row[key], indices=key in {"slot", "sites"})
                    for key in headers
                    if key in row
                ]
            )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    return "\n".join(
        "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    )
