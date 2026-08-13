from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tinycable.core.morphology import Morphology


def plot_morphology(
    morphology: "Morphology",
    *,
    dims: str = "xy",
    ax: Any | None = None,
    marker: str | None = None,
    kind: str = "swc",
) -> Any:
    """Plot segmented compartments or the raw SWC morphology."""
    assert len(dims) == 2 and all(dim in "xyz" for dim in dims), (
        "dims must contain two axes from 'xyz'"
    )
    assert dims[0] != dims[1], "dims must contain two distinct axes"
    assert kind in {"seg", "swc"}, "kind must be 'seg' or 'swc'"

    if kind == "seg":
        if morphology.xyz.shape != (morphology.n, 3):
            raise ValueError("Morphology.xyz must have shape (n, 3) for plotting")
        points = morphology.xyz
        edges = np.flatnonzero(morphology.tree != morphology.index)
        parents = morphology.tree[edges]
    else:
        swc = np.asarray(morphology.swc)
        points = swc[:, 2:5]
        ids = swc[:, 0].astype(np.int64)
        rows = {int(node_id): row for row, node_id in enumerate(ids)}
        assert len(rows) == len(ids), "SWC node IDs must be unique"
        raw_parents = swc[:, 6].astype(np.int64)
        edges = np.flatnonzero(raw_parents >= 0)
        try:
            parents = np.asarray(
                [rows[int(parent)] for parent in raw_parents[edges]],
                dtype=np.int32,
            )
        except KeyError as error:
            raise ValueError("SWC parent ID does not reference a node") from error

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    axes = {"x": 0, "y": 1, "z": 2}
    x_axis, y_axis = axes[dims[0]], axes[dims[1]]
    if len(edges):
        x = np.full(3 * len(edges), np.nan)
        y = np.full(3 * len(edges), np.nan)
        x[0::3] = points[parents, x_axis]
        x[1::3] = points[edges, x_axis]
        y[0::3] = points[parents, y_axis]
        y[1::3] = points[edges, y_axis]
        ax.plot(x, y)
    if marker is not None:
        ax.plot(
            points[:, x_axis],
            points[:, y_axis],
            linestyle="None",
            marker=marker,
        )
    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])
    ax.set_aspect("equal")
    return ax
