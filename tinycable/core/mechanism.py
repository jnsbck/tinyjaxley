from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from .field import Field
from .utils import Ns, assert_index


Declaration = str | Field


@dataclass(frozen=True, eq=False)
class Mechanism:
    """Pure state, parameter, derivative, and current declarations."""

    name: str | None = None
    index: npt.ArrayLike | None = field(default=None, compare=False)
    currents: ClassVar[tuple[str, ...]] = ()
    states: ClassVar[tuple[Declaration, ...]] = ()
    params: ClassVar[tuple[Declaration, ...]] = ()
    density: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", type(self).__name__.lower())
        if self.index is not None:
            index = assert_index(self.index)
            object.__setattr__(self, "index", index)

    def declarations(self) -> Iterator[tuple[str, str, Field | None]]:
        for namespace, declarations in (("s", self.states), ("p", self.params)):
            for declaration in declarations:
                if isinstance(declaration, str):
                    yield namespace, declaration, None
                    continue
                field = declaration
                if self.index is not None:
                    field = field._insert(
                        self.index,
                        prefix=self.name,
                        dynamic=namespace == "s",
                    )
                yield namespace, field.key, field

    def d(self, t: Any, s: Ns, p: Ns) -> dict[str, Any]:
        return {}

    def i(self, t: Any, s: Ns, p: Ns) -> Any:
        return {}


class Channel(Mechanism):
    density: ClassVar[bool] = True


@dataclass(frozen=True, eq=False, init=False)
class Synapse(Mechanism):
    """Pure declaration for a directed edge mechanism."""

    pre_index: npt.ArrayLike | None = None
    post_index: npt.ArrayLike | None = None
    pre: ClassVar[tuple[str, ...]] = ()
    post: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        pre_index: npt.ArrayLike | None = None,
        post_index: npt.ArrayLike | None = None,
        name: str | None = None,
        index: npt.ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "pre_index", self._endpoint_array(pre_index))
        object.__setattr__(self, "post_index", self._endpoint_array(post_index))
        Mechanism.__post_init__(self)
        if self.pre_index is not None and self.post_index is not None:
            assert len(self.pre_index) == len(self.post_index), (
                "pre_index and post_index must have equal lengths"
            )

    @staticmethod
    def _endpoint_array(values: npt.ArrayLike | None) -> np.ndarray | None:
        if values is None:
            return None
        return assert_index(values, sorted=False)

    def d(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> dict[str, Any]:
        return {}

    def i(self, t: Any, s: Ns, p: Ns, pre: Ns, post: Ns) -> Any:
        return {}
