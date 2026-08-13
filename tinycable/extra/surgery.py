from dataclasses import replace as replace_dataclass
from typing import Any, Callable

import jax

from tinycable.core.runtime import Access, BoundMechanism, FrozenMap, Runtime


_MISSING = object()
_EDITABLE = {
    Runtime: ("pool", "mechs", "syns", "morph"),
    BoundMechanism: ("sink", "ns"),
    Access: ("index",),
}


def surgery(
    runtime: Runtime,
    path: str | tuple[str, ...],
    *,
    replace: Any = _MISSING,
    replace_fn: Callable[[Any], Any] | None = None,
) -> Runtime:
    """Experimentally replace one same-shaped Runtime array leaf.

    Pool paths edit parameter or state values; route paths edit executable
    projections. Both operations bypass semantic Runtime validation.
    """
    if (replace is _MISSING) == (replace_fn is None):
        raise TypeError("surgery requires exactly one of replace or replace_fn")
    if replace_fn is not None and not callable(replace_fn):
        raise TypeError("replace_fn must be callable")
    if isinstance(path, str):
        parts = tuple(path.split("."))
    elif isinstance(path, tuple) and all(isinstance(part, str) for part in path):
        parts = path
    else:
        raise TypeError("surgery path must be a dotted string or tuple of strings")
    if not parts or not all(parts):
        raise ValueError("surgery path segments must be non-empty")
    label = ".".join(parts)

    def update(value: Any) -> Any:
        if value is None:
            raise ValueError(f"surgery path {label!r} selects IDENTITY")
        if not isinstance(value, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"surgery path {label!r} must select an array leaf")
        result = replace_fn(value) if replace_fn is not None else replace
        if not isinstance(result, (jax.Array, jax.core.Tracer)):
            raise TypeError("surgery replacement must be an array leaf")
        if result.shape != value.shape:
            raise ValueError(
                f"surgery replacement shape {result.shape} differs from {value.shape}"
            )
        if result.dtype != value.dtype:
            raise ValueError(
                f"surgery replacement dtype {result.dtype} differs from {value.dtype}"
            )
        return result

    def rewrite(node: Any, remaining: tuple[str, ...]) -> Any:
        if not remaining:
            return update(node)
        part, rest = remaining[0], remaining[1:]
        if isinstance(node, FrozenMap):
            if part not in node:
                raise KeyError(f"unknown surgery path {label!r}")
            values = dict(node)
            values[part] = rewrite(node[part], rest)
            return FrozenMap(values)
        for kind, editable in _EDITABLE.items():
            if isinstance(node, kind):
                if part not in editable:
                    if hasattr(node, part):
                        raise TypeError(f"{kind.__name__}.{part} is static metadata")
                    raise KeyError(f"unknown surgery path {label!r}")
                return replace_dataclass(
                    node,
                    **{part: rewrite(getattr(node, part), rest)},
                )
        raise KeyError(f"unknown surgery path {label!r}")

    return rewrite(runtime, parts)
