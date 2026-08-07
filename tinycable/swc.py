import numpy as np


_SWC_TYPE_NAMES = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal",
    4: "apical",
    5: "fork",
    6: "end",
    7: "custom",
}


def section_tree(tree):
    root = tree == np.arange(len(tree))
    nchild = np.bincount(tree[~root], minlength=len(tree))
    secs = []

    for n in np.flatnonzero((nchild != 1) & ~root):
        sec = [n]
        while not root[n] and nchild[tree[n]] == 1:
            n = tree[n]
            sec.append(n)
        if not root[n]:
            sec.append(tree[n])
        secs.append(np.array(sec[::-1]))
    return secs


def _seg_attrs(xyzr, nseg):
    """x, y, z, r, len, area, volume, rin, rout."""
    d = np.linalg.norm(np.diff(xyzr[:, :3], axis=0), axis=1)
    assert np.all(d > 0), "consecutive SWC points must be distinct"

    s = np.r_[0, d.cumsum()]
    L = s[-1] / nseg

    # Include SWC points and half-compartment boundaries.
    b = np.linspace(0, s[-1], 2 * nseg + 1)
    x = np.unique(np.r_[s, b])
    v = np.array([np.interp(x, s, a) for a in xyzr.T]).T

    dl = np.diff(x)
    r0, r1 = v[:-1, 3], v[1:, 3]
    assert np.all((r0 > 0) & (r1 > 0)), "SWC radii must be positive"

    h = np.searchsorted(b, (x[:-1] + x[1:]) / 2, side="right") - 1

    def sum_(a):
        return np.bincount(h, weights=a, minlength=2 * nseg).reshape(nseg, 2)

    rad = sum_(dl * (r0 + r1) / 2).sum(1) / L
    area = sum_(np.pi * (r0 + r1) * np.hypot(dl, r1 - r0)).sum(1) * 1e-8
    volume = sum_(np.pi / 3 * dl * (r0 * r0 + r0 * r1 + r1 * r1)).sum(1)
    load = sum_(dl / (np.pi * r0 * r1))

    mid = (np.arange(nseg) + 0.5) * L
    xyz = np.array([np.interp(mid, s, a) for a in xyzr[:, :3].T]).T

    return np.c_[xyz, rad, np.full(nseg, L), area, volume, load]


def segment_swc(
    swc, nseg_per_sec: int = 4
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    swc = np.asarray(swc, dtype=float)
    assert swc.ndim == 2 and swc.shape[1] == 7, "SWC must have shape (n, 7)"
    assert nseg_per_sec > 0, "nseg_per_sec must be positive"

    n = len(swc)
    if not n:
        return np.empty(0, np.int32), np.empty((0, 9)), {}
    assert np.array_equal(swc[:, 0], np.arange(1, n + 1)), (
        "SWC ids must be sequential from 1"
    )

    swc_tree = swc[:, 6].astype(np.int32) - 1
    roots = swc_tree < 0
    swc_tree[roots] = np.flatnonzero(roots)
    secs = section_tree(swc_tree)

    nchild = np.bincount(swc_tree[~roots], minlength=n)
    junctions = np.flatnonzero(roots | (nchild > 1))
    nj = len(junctions)

    junction_of = np.full(n, -1, np.int32)
    junction_of[junctions] = np.arange(nj)

    sec_segs = nj + np.arange(len(secs) * nseg_per_sec, dtype=np.int32).reshape(
        -1, nseg_per_sec
    )

    tree = np.arange(nj + sec_segs.size, dtype=np.int32)
    starts = np.array([s[0] for s in secs])
    ends = np.array([s[-1] for s in secs])

    tree[sec_segs[:, 0]] = junction_of[starts]
    tree[sec_segs[:, 1:]] = sec_segs[:, :-1]

    j = junction_of[ends]
    mask = j >= 0
    tree[j[mask]] = sec_segs[mask, -1]

    # x, y, z, r, len, area, volume, rin, rout
    attrs = np.zeros((len(tree), 9))
    attrs[:nj, :4] = swc[junctions, 2:6]

    for sec, segs in zip(secs, sec_segs):
        attrs[segs] = _seg_attrs(swc[sec, 2:6], nseg_per_sec)

    labels = {
        name: np.zeros(len(tree), dtype=bool) for name in _SWC_TYPE_NAMES.values()
    }
    types = swc[:, 1].astype(np.int32)
    for compartment, point in enumerate(junctions):
        name = _SWC_TYPE_NAMES.get(int(types[point]), f"type_{types[point]}")
        labels.setdefault(name, np.zeros(len(tree), dtype=bool))[compartment] = True
    for sec, segs in zip(secs, sec_segs):
        point = sec[1] if len(sec) > 1 else sec[0]
        name = _SWC_TYPE_NAMES.get(int(types[point]), f"type_{types[point]}")
        labels.setdefault(name, np.zeros(len(tree), dtype=bool))[segs] = True
    return tree, attrs, labels