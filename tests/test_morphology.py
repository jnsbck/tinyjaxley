import numpy as np
import pytest

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


def test_morphology_labels_are_derived_and_immutable():
    morphology = Morphology([0, 0, 1, 2, 2])

    # Index identifies rows; cells and branches are derived from the tree.
    np.testing.assert_array_equal(morphology.index, np.array([0, 1, 2, 3, 4], np.int32))
    np.testing.assert_array_equal(morphology.labels["comp"], morphology.index)
    np.testing.assert_array_equal(
        morphology.labels["branch"], np.array([0, 0, 0, 1, 2], np.int32)
    )
    np.testing.assert_array_equal(
        morphology.labels["cell"], np.zeros(5, dtype=np.int32)
    )
    assert all(
        not values.flags.writeable
        for values in (
            morphology.index,
            *morphology.labels.values(),
        )
    )

    with pytest.raises(TypeError):
        morphology.labels["soma"] = np.array([True, False, False, False, False])


def test_morphology_accepts_custom_labels():
    morphology = Morphology(
        [0, 0, 1],
        labels={
            "comp": [0, 1, 2],
            "branch": [0, 0, 1],
            "cell": [0, 0, 1],
            "soma": [True, False, False],
        },
    )

    np.testing.assert_array_equal(morphology.labels["comp"], [0, 1, 2])
    np.testing.assert_array_equal(morphology.labels["branch"], [0, 0, 1])
    np.testing.assert_array_equal(morphology.labels["cell"], [0, 0, 1])
    np.testing.assert_array_equal(morphology.labels["soma"], [True, False, False])

    empty = Morphology([], labels={"comp": np.empty(0, dtype=np.int32)})
    assert empty.labels["comp"].shape == (0,)

    with pytest.raises(AssertionError, match="boolean-valued"):
        Morphology([0], labels={"soma": [1]})
    with pytest.raises(AssertionError, match="soma must have length 3"):
        Morphology([0, 0, 1], labels={"soma": [True]})


def test_morphology_label_defaults_follow_compartment_rows():
    morphology = Cable(3)

    np.testing.assert_array_equal(morphology.index, np.arange(3, dtype=np.int32))
    np.testing.assert_array_equal(morphology.labels["comp"], morphology.index)
    np.testing.assert_array_equal(morphology.labels["branch"], np.zeros(3, np.int32))
    np.testing.assert_array_equal(morphology.labels["cell"], np.zeros(3, np.int32))


def test_singleton_roots_receive_independent_branch_labels():
    morphology = Morphology([0, 1])

    np.testing.assert_array_equal(morphology.labels["comp"], [0, 1])
    np.testing.assert_array_equal(morphology.labels["branch"], [0, 1])
    np.testing.assert_array_equal(morphology.labels["cell"], [0, 1])


def test_from_swc_preserves_type_labels_and_resistive_loads():
    swc = np.array(
        [
            [1, 1, 0.0, 0.0, 0.0, 2.0, -1],
            [2, 3, 1.0, 0.0, 0.0, 1.5, 1],
            [3, 4, 2.0, 0.0, 0.0, 1.0, 2],
        ]
    )
    morphology = Morphology.from_swc(swc, nseg_per_sec=1)

    assert morphology.labels["soma"].any()
    assert morphology.labels["basal"].any()
    assert not morphology.labels["apical"].any()
    assert morphology.rin.shape == (morphology.n,)
    assert morphology.rout.shape == (morphology.n,)


def test_morphology_extend_appends_a_disjoint_cell():
    left = Cable(2, comp_len=10.0, comp_rad=2.0)
    right = Morphology(
        [0, 0],
        len=[3.0, 4.0],
        rad=[1.0, 1.5],
        xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        swc=Cable(2).swc,
        labels={"soma": [True, False]},
    )

    merged = left.extend(right)

    # The second morphology becomes a separate rooted cell with offset rows.
    np.testing.assert_array_equal(merged.tree, np.array([0, 0, 2, 2], np.int32))
    np.testing.assert_array_equal(merged.index, np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(merged.labels["comp"], [0, 1, 2, 3])
    np.testing.assert_array_equal(merged.labels["branch"], [0, 0, 1, 1])
    np.testing.assert_array_equal(merged.labels["cell"], [0, 0, 1, 1])
    np.testing.assert_array_equal(merged.labels["soma"], [False, False, True, False])
    np.testing.assert_array_equal(merged.len, [10.0, 10.0, 3.0, 4.0])
    np.testing.assert_array_equal(merged.rad, [2.0, 2.0, 1.0, 1.5])
    np.testing.assert_array_equal(
        merged.xyz,
        [[5.0, 0.0, 0.0], [15.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    np.testing.assert_array_equal(merged.swc[:, 0], np.arange(1.0, 7.0))
    np.testing.assert_array_equal(merged.swc[:, 6], [-1.0, 1.0, 2.0, -1.0, 4.0, 5.0])


def test_morphology_plot_plots_segmented_edges_and_optional_nodes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    morphology = Cable(3)
    ax = morphology.plot(dims="xz", marker="o")

    # Parent-child edges use NaN breaks; markers are a separate node artist.
    assert len(ax.lines) == 2
    np.testing.assert_array_equal(
        ax.lines[0].get_xdata(), [5.0, 15.0, np.nan, 15.0, 25.0, np.nan]
    )
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [5.0, 15.0, 25.0])
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "z"
    plt.close(ax.figure)


def test_morphology_plot_can_show_raw_swc():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    morphology = Cable(2)
    ax = morphology.plot(dims="xy", kind="swc", marker=".")

    # Raw SWC coordinates and parent IDs produce the full source tree.
    assert len(ax.lines) == 2
    np.testing.assert_array_equal(
        ax.lines[0].get_xdata(), [0.0, 10.0, np.nan, 10.0, 20.0, np.nan]
    )
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [0.0, 10.0, 20.0])
    plt.close(ax.figure)
