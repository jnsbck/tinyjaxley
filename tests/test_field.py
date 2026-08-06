import numpy as np
import pytest

from tinycable import Field


def test_field_normalizes_exact_slots():
    sites = np.array([0, 2], dtype=np.int32)
    scalar = Field("voltage", -65.0, dynamic=True, index=sites)
    per_slot = Field("rad", [1.0, 2.0], index=sites)
    vector_default = np.array([0.1, 0.2], dtype=np.float32)
    vector = Field("weights", vector_default).place(sites)

    # Placed Fields materialize exact slots and explicit site-to-slot groups.
    assert scalar.dynamic
    assert scalar.n_slots == 2
    assert scalar.value_shape == ()
    np.testing.assert_array_equal(scalar.slots, np.array([-65.0, -65.0]))
    np.testing.assert_array_equal(scalar.groups, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(per_slot.slots, np.array([1.0, 2.0]))

    # An unplaced array is one vector slot that placement broadcasts over sites.
    assert vector.value_shape == (2,)
    assert vector.slots.shape == (2, 2)
    assert vector.slots.dtype == np.float32
    np.testing.assert_array_equal(
        vector.slots, np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)
    )

    # Declaration arrays are immutable copies of caller-owned inputs.
    sites[0] = 9
    vector_default[0] = 9.0
    np.testing.assert_array_equal(vector.index, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        vector.slots[0], np.array([0.1, 0.2], dtype=np.float32)
    )
    assert not vector.index.flags.writeable
    assert not vector.groups.flags.writeable
    assert not vector.slots.flags.writeable


def test_unplaced_field_is_one_slot_template():
    scalar = Field("temperature", 37.0)
    vector = Field("weights", [0.1, 0.2])

    # Unplaced templates expose one slot but have no support or grouping yet.
    assert scalar.index is None
    assert scalar.groups is None
    assert scalar.n_slots == 1
    assert scalar.value_shape == ()
    np.testing.assert_array_equal(scalar.values, np.array([37.0]))
    assert vector.slots.shape == (1, 2)
    assert vector.value_shape == (2,)

    # Support-dependent operations require placement.
    with pytest.raises(ValueError, match="must be placed"):
        scalar.slot_index([0])
    with pytest.raises(ValueError, match="must be placed"):
        scalar.extend(Field("temperature", 37.0, index=[0]))
    with pytest.raises(ValueError, match="already placed"):
        scalar.place([0]).place([0])
    with pytest.raises(AssertionError, match="unplaced fields cannot define groups"):
        Field("temperature", 37.0, groups=[0])


def test_slot_index_validates_support():
    field = Field("conductance", [0.1, 0.2, 0.3], index=[1, 3, 5])

    # Projection returns raw int32 slot indices in requested-site order.
    np.testing.assert_array_equal(
        field.slot_index([1, 3, 5]), np.array([0, 1, 2], dtype=np.int32)
    )
    index = field.slot_index([5, 1])
    assert index.dtype == np.int32
    np.testing.assert_array_equal(index, np.array([2, 0], dtype=np.int32))

    # Missing indices fail while fractional values follow NumPy int32 conversion.
    with pytest.raises(ValueError, match=r"indices \[2\]"):
        field.slot_index([1, 2])
    np.testing.assert_array_equal(
        field.slot_index([1.5]), np.array([0], dtype=np.int32)
    )
    with pytest.raises(AssertionError, match="sorted and deduplicated"):
        Field("unsorted", [1.0, 2.0], index=[3, 1])
    with pytest.raises(AssertionError, match="sorted and deduplicated"):
        Field("duplicate", [1.0, 2.0], index=[1, 1])


def test_group_storage_and_values():
    groups = np.array([0, 0, 1, 1], dtype=np.int32)
    grouped = Field("g", [10.0, 20.0], index=[0, 1, 2, 3], groups=groups)
    groups[0] = 1

    # Slots are storage values; values are their projection over support order.
    assert grouped.n_slots == 2
    assert grouped.value_shape == ()
    np.testing.assert_array_equal(grouped.slots, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(grouped.groups, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(grouped.values, np.array([10.0, 10.0, 20.0, 20.0]))
    np.testing.assert_array_equal(
        grouped.slot_index(grouped.index), np.array([0, 0, 1, 1], dtype=np.int32)
    )

    # One vector slot can be materialized behind an arbitrary support grouping.
    vector = Field(
        "weights",
        [[0.1, 0.2]],
        index=[0, 1, 2],
        groups=[0, 0, 0],
    )
    assert vector.slots.shape == (1, 2)
    assert vector.values.shape == (3, 2)

    # Placed groups are explicit, aligned, compact integer arrays.
    with pytest.raises(AssertionError, match="one-dimensional"):
        Field("g", 1.0, index=[0, 1], groups=0)
    with pytest.raises(AssertionError, match="compact from zero"):
        Field("g", [1.0, 2.0], index=[0, 1], groups=[0, 2])
    with pytest.raises(AssertionError, match="align with index"):
        Field("g", 1.0, index=[0, 1], groups=[0])
    float_groups = Field("g", [1.0, 2.0], index=[0, 1], groups=[0.0, 1.0])
    np.testing.assert_array_equal(float_groups.groups, np.array([0, 1], dtype=np.int32))
    with pytest.raises(ValueError, match="slot values for 2 groups"):
        Field("g", [1.0, 2.0, 3.0], index=[0, 1], groups=[0, 1])


def test_regroup_reduces_current_values():
    field = Field("g", [10.0, 20.0, 30.0, 40.0], index=range(4))

    # Regrouping defaults to the first support value in each destination group.
    grouped = field.regroup([0, 0, 1, 1])
    np.testing.assert_array_equal(grouped.slots, np.array([10.0, 30.0]))
    np.testing.assert_array_equal(grouped.values, np.array([10.0, 10.0, 30.0, 30.0]))

    # Repeated regrouping projects current slots before applying a custom reducer.
    regrouped = grouped.regroup([0, 1, 1, 2], reduce=lambda values: values.mean(axis=0))
    np.testing.assert_array_equal(regrouped.slots, np.array([10.0, 20.0, 30.0]))

    # Reduction acts only on support axis zero and preserves value shape.
    vector = Field(
        "weights",
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        index=range(4),
    ).regroup([0, 0, 1, 1], reduce=lambda values: values.mean(axis=0))
    assert vector.slots.shape == (2, 2)
    np.testing.assert_array_equal(vector.slots, np.array([[2.0, 3.0], [6.0, 7.0]]))

    # Empty support creates no slot and never invokes the reducer.
    def fail(_values):
        raise AssertionError("empty regrouping called its reducer")

    empty = Field("empty", 0.0).place([]).regroup([], reduce=fail)
    assert empty.n_slots == 0
    assert empty.slots.shape == (0,)
    assert empty.values.shape == (0,)


def test_extend_merges_support_slots_and_groups():
    left = Field("g", [10.0, 20.0], index=[0, 2])
    right = Field("g", [30.0, 40.0], dynamic=True, index=[1, 2])

    # Right-biased union merges independent support values and dynamic status.
    merged = left.extend(right)
    np.testing.assert_array_equal(merged.index, np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(merged.values, np.array([10.0, 30.0, 40.0]))
    assert merged.dynamic

    # Callers can preserve the left declaration on overlap.
    left_owned = left.extend(right, left_wins=True)
    np.testing.assert_array_equal(left_owned.values, np.array([10.0, 30.0, 20.0]))

    # Existing and incoming source-slot ties remain independently namespaced.
    shared = Field("g", 10.0, index=[0, 1], groups=[0, 0])
    incoming = Field("g", 40.0, index=[1, 2], groups=[0, 0])
    right_shared = shared.extend(incoming)
    np.testing.assert_array_equal(right_shared.values, np.array([10.0, 40.0, 40.0]))
    assert right_shared.groups[1] == right_shared.groups[2]
    assert right_shared.groups[0] != right_shared.groups[1]

    # Losing overlap can split a source group without separating its survivors.
    broad = Field("g", 10.0, index=[0, 1, 2], groups=[0, 0, 0])
    split = broad.extend(Field("g", 20.0, index=[1]))
    np.testing.assert_array_equal(split.values, np.array([10.0, 20.0, 10.0]))
    assert split.groups[0] == split.groups[2] != split.groups[1]

    # Empty support contributes no values but still propagates dynamic status.
    empty = Field("g", 0.0, dynamic=True).place([])
    extended = left.extend(empty)
    np.testing.assert_array_equal(extended.values, left.values)
    assert extended.dynamic

    with pytest.raises(AssertionError, match="same name"):
        left.extend(Field("other", 0.0, index=[0]))
    with pytest.raises(AssertionError, match="same value shape"):
        left.extend(Field("g", [[1.0, 2.0]], index=[0]))
