import numpy as np

from tinycable import Cable, Morphology, Point


def test_morphology_construction():
    empty = Morphology([])
    single = Point(comp_len=20.0, comp_rad=2.0)
    raw_swc = np.arange(14.0).reshape(2, 7)
    archived = Morphology([], swc=raw_swc)

    # Empty and one-compartment morphologies share array-based storage.
    assert empty.n == 0
    assert empty.len.shape == (0,)
    assert empty.rad.shape == (0,)
    assert empty.xyz.shape == (0, 3)
    assert empty.swc.shape == (0, 7)
    assert single.n == 1
    assert isinstance(single, Cable)
    assert isinstance(single, Morphology)
    assert single.tree.dtype == np.int32
    np.testing.assert_array_equal(single.tree, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(single.len, np.array([20.0]))
    np.testing.assert_array_equal(single.rad, np.array([2.0]))
    np.testing.assert_array_equal(single.xyz, np.array([[10.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(
        single.swc,
        np.array(
            [
                [1.0, 3.0, 0.0, 0.0, 0.0, 2.0, -1.0],
                [2.0, 3.0, 20.0, 0.0, 0.0, 2.0, 1.0],
            ]
        ),
    )

    # Raw SWC source data is retained independently of compartmentalization.
    assert archived.n == 0
    np.testing.assert_array_equal(archived.swc, raw_swc)

    # Topology and initial geometry cannot be mutated after construction.
    assert not empty.tree.flags.writeable
    assert not single.tree.flags.writeable
    assert not single.len.flags.writeable
    assert not single.rad.flags.writeable
    assert not single.xyz.flags.writeable
    assert not single.swc.flags.writeable

    # Area uses simulator cm^2 units; volume remains in morphology um^3 units.
    np.testing.assert_allclose(single.area, 2.0 * np.pi * 2.0 * 20.0 * 1e-8)
    np.testing.assert_allclose(single.volume, np.pi * 2.0**2 * 20.0)


def test_cable_morphology():
    cable = Cable(4, comp_len=10.0, comp_rad=2.0)

    # Compartments form an unbranched parent-pointer tree with a self-rooted root.
    np.testing.assert_array_equal(cable.tree, np.array([0, 0, 1, 2], np.int32))
    np.testing.assert_array_equal(cable.len, np.full(4, 10.0))
    np.testing.assert_array_equal(cable.rad, np.full(4, 2.0))

    # Compartment centers lie halfway between the retained SWC endpoints.
    np.testing.assert_array_equal(cable.xyz[:, 0], np.array([5.0, 15.0, 25.0, 35.0]))
    np.testing.assert_array_equal(cable.xyz[:, 1:], np.zeros((4, 2)))
    assert cable.swc.shape == (5, 7)
    np.testing.assert_array_equal(cable.swc[:, 2], np.arange(0.0, 50.0, 10.0))
    np.testing.assert_array_equal(cable.swc[:, 6], np.array([-1.0, 1.0, 2.0, 3.0, 4.0]))

    # Derived geometry remains compartment-sized.
    assert cable.area.shape == (4,)
    assert cable.volume.shape == (4,)
