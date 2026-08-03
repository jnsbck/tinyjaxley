import numpy as np

from tinycable import Morphology


def test_morphology_construction():
    empty = Morphology([])
    single = Morphology.single(len=20.0, rad=2.0, xyz=(1.0, 2.0, 3.0))
    raw_swc = np.arange(14.0).reshape(2, 7)
    archived = Morphology([], swc=raw_swc)

    # Empty and single-compartment morphologies share array-based storage.
    assert empty.n == 0
    assert empty.len.shape == (0,)
    assert empty.rad.shape == (0,)
    assert empty.xyz.shape == (0, 3)
    assert empty.swc.shape == (0, 7)
    assert single.n == 1
    assert single.parent.dtype == np.int32
    np.testing.assert_array_equal(single.parent, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(single.len, np.array([20.0]))
    np.testing.assert_array_equal(single.rad, np.array([2.0]))
    np.testing.assert_array_equal(single.xyz, np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_array_equal(
        single.swc,
        np.array([[1.0, 1.0, 1.0, 2.0, 3.0, 2.0, -1.0]]),
    )

    # Raw SWC source data is retained independently of compartmentalization.
    assert archived.n == 0
    np.testing.assert_array_equal(archived.swc, raw_swc)

    # Topology and initial geometry cannot be mutated after construction.
    assert not empty.parent.flags.writeable
    assert not single.parent.flags.writeable
    assert not single.len.flags.writeable
    assert not single.rad.flags.writeable
    assert not single.xyz.flags.writeable
    assert not single.swc.flags.writeable

    # Area uses simulator cm^2 units; volume remains in morphology um^3 units.
    np.testing.assert_allclose(single.area, 2.0 * np.pi * 2.0 * 20.0 * 1e-8)
    np.testing.assert_allclose(single.volume, np.pi * 2.0**2 * 20.0)
