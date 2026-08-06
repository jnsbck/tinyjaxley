import numpy as np
import pytest

from tinycable import K, Na, Cable, Channel, Field, Leak, Model, Morphology


def _project(field):
    return field.values


def _fields(model):
    return model._fields


def test_model_base_fields_are_compartment_scalars():
    morphology = Cable(3)
    voltage = np.array([-70.0, -65.0, -60.0])
    axial_resistivity = np.array([100.0, 110.0, 120.0])
    capacitance = np.array([0.8, 0.9, 1.0])
    model = Model(morphology, v=voltage, Ra=axial_resistivity, cm=capacitance)
    fields = _fields(model)

    # Array constructor values describe slots, not trailing vector payloads.
    for name, expected in (
        ("v", voltage),
        ("Ra", axial_resistivity),
        ("cm", capacitance),
    ):
        assert fields[name].value_shape == ()
        assert fields[name].slots.shape == (3,)
        np.testing.assert_array_equal(fields[name].slots, expected)

    with pytest.raises(AssertionError, match="base fields must be scalar-valued"):
        Model(morphology, v=[[-70.0, -65.0, -60.0]])


def test_field_insertion_supplies_support_and_preserves_groups():
    unplaced = Field("temperature", 37.0)
    shared = Field("gain", 2.0)
    vector = Field("weights", [0.1, 0.2])
    placed = Model(Cable(3)).insert(unplaced).insert(shared).insert(vector)
    model = placed.share("gain")

    # Unplaced Fields become independent slots over all model compartments.
    all_sites = np.arange(3, dtype=np.int32)
    np.testing.assert_array_equal(model._fields["temperature"].index, all_sites)
    assert model._fields["temperature"].n_slots == 3
    np.testing.assert_array_equal(model._fields["temperature"].slots, np.full(3, 37.0))

    # Scalar grouping broadcasts over placement but allocates only final slots.
    np.testing.assert_array_equal(model._fields["gain"].index, all_sites)
    np.testing.assert_array_equal(model._fields["gain"].groups, np.zeros(3))
    assert model._fields["gain"].n_slots == 1
    np.testing.assert_array_equal(model._fields["gain"].slots, np.array([2.0]))
    assert model._fields["weights"].slots.shape == (3, 2)

    # Functional placement leaves source declarations unplaced and immutable.
    assert unplaced.index is None
    assert shared.index is None
    assert shared.groups is None

    # Empty morphology placement creates no synthetic storage slot.
    empty = Model(Morphology([])).insert(Field("empty", 1.0))
    assert empty._fields["empty"].n_slots == 0
    assert empty._fields["empty"].slots.shape == (0,)


def test_insert_places_fields_and_mechanisms_on_a_cable():
    morphology = Cable(4, comp_len=10.0, comp_rad=2.0)
    base = Model(morphology, v=-70.0, Ra=150.0, cm=0.8)
    temperature = Field("temperature", 37.0, index=[1, 3])
    model = (
        base.insert(temperature)
        .insert(Na(name="sodium", index=[1, 2, 3]))
        .insert(K(index=[0, 2]))
    )
    fields = _fields(model)

    # Model-owned electrical and geometry fields cover the complete cable.
    all_sites = np.arange(4, dtype=np.int32)
    assert tuple(_fields(base)) == ("v", "Ra", "cm", "rad", "len")
    for name in ("v", "Ra", "cm", "rad", "len"):
        np.testing.assert_array_equal(fields[name].index, all_sites)
    np.testing.assert_array_equal(fields["rad"].slots, morphology.rad)
    np.testing.assert_array_equal(fields["len"].slots, morphology.len)
    assert fields["v"].dynamic

    # Field and Mechanism placement applies to declared states and parameters.
    np.testing.assert_array_equal(fields["temperature"].index, np.array([1, 3]))
    for name in ("Na.m", "Na.h", "Na.g", "eNa"):
        np.testing.assert_array_equal(fields[name].index, np.array([1, 2, 3]))
    for name in ("K.n", "K.g", "eK"):
        np.testing.assert_array_equal(fields[name].index, np.array([0, 2]))
    assert all(fields[name].dynamic for name in ("v", "Na.m", "Na.h", "K.n"))

    # Resulting supports provide identity and indexed access paths as appropriate.
    np.testing.assert_array_equal(
        fields["Na.m"].slot_index([1, 2, 3]),
        np.array([0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        fields["v"].slot_index([1, 2, 3]),
        np.array([1, 2, 3], dtype=np.int32),
    )

    # Unplaced mechanisms expand to all compartments when inserted.
    all_sites_model = base.insert(Leak())
    np.testing.assert_array_equal(all_sites_model._mechs["leak"].index, all_sites)

    # Placement edits are functional and declarations retain immutable site arrays.
    assert base._mechs == {}
    assert tuple(model._mechs) == ("sodium", "k")
    assert all(
        not mechanism.index.flags.writeable for mechanism in model._mechs.values()
    )

    # Repeated Field insertion extends the value stored under its field name.
    extended = base.insert(Field("temperature", 36.0, index=[0])).insert(
        Field("temperature", 37.0, index=[2])
    )
    np.testing.assert_array_equal(
        extended._fields["temperature"].index, np.array([0, 2])
    )
    np.testing.assert_array_equal(
        extended._fields["temperature"].slots, np.array([36.0, 37.0])
    )

    # Frozen Models expose inspectable but read-only declaration mappings.
    with pytest.raises(TypeError):
        model._fields["other"] = Field("other", 0.0)


def test_insert_replaces_and_remove_keeps_dangling_fields():
    base = Model(Cable(4))
    model = (
        base.insert(Field("eNa", 55.0, index=[1, 2, 3]))
        .insert(Field("temperature", 37.0, index=[0, 3]))
        .insert(Na(name="channel", index=[1, 2, 3]))
        .insert(K(index=[0, 2]))
    )

    # Replacing a named mechanism updates the mechanism but leaves earlier fields for now.
    replaced = model.insert(Leak(name="channel", index=[3]))
    assert tuple(replaced._mechs) == ("channel", "k")
    assert isinstance(replaced._mechs["channel"], Leak)
    assert "Na.m" in _fields(replaced)
    np.testing.assert_array_equal(
        _fields(replaced)["Leak.g"].index, np.array([1, 2, 3])
    )

    # Removal warns and removes only the mechanism while ownership remains unresolved.
    with pytest.warns(UserWarning, match="leaves its declared fields"):
        removed = model.remove("channel")
    assert "channel" not in removed._mechs
    assert "Na.m" in _fields(removed)
    assert "Na.g" in _fields(removed)
    np.testing.assert_array_equal(_fields(removed)["eNa"].index, np.array([1, 2, 3]))
    assert "K.n" in _fields(removed)

    # Unknown names and Field names are no-ops until Field removal is defined.
    assert removed.remove("temperature") is removed
    assert removed.remove("missing") is removed

    # Earlier model values remain unchanged after every functional edit.
    assert "channel" in model._mechs
    assert "temperature" in model._fields

    # Field and Mechanism names coexist in independent dictionaries.
    collision = base.insert(Field("channel", 37.0)).insert(
        Na(name="channel", index=[1, 2])
    )
    assert "channel" in collision._fields
    assert "channel" in collision._mechs
    with pytest.warns(UserWarning):
        cleared = collision.remove("channel")
    assert "channel" in cleared._fields
    assert "channel" not in cleared._mechs


def test_repeated_mechanism_insertion_unions_placement():
    base = Model(Cable(5))
    model = base.insert(Leak(index=[1, 2])).insert(Leak(index=[4]))

    # Repeated insertion of one named mechanism creates one combined instance set.
    assert tuple(model._mechs) == ("leak",)
    assert isinstance(model._mechs["leak"], Leak)
    np.testing.assert_array_equal(model._mechs["leak"].index, np.array([1, 2, 4]))

    # Every field declared by that mechanism receives the combined support.
    fields = _fields(model)
    np.testing.assert_array_equal(fields["Leak.g"].index, np.array([1, 2, 4]))
    np.testing.assert_array_equal(fields["eL"].index, np.array([1, 2, 4]))


def test_share_is_eager_and_unshare_uses_current_values():
    base = Model(Cable(4)).insert(Field("g", [10.0, 20.0, 30.0, 40.0], index=range(4)))

    # Sharing immediately replaces the Field only in the returned Model.
    shared = base.share("g", groups=[0, 0, 1, 1])
    np.testing.assert_array_equal(base._fields["g"].groups, np.arange(4))
    np.testing.assert_array_equal(shared._fields["g"].slots, np.array([10.0, 30.0]))
    np.testing.assert_array_equal(
        _project(shared._fields["g"]), np.array([10.0, 10.0, 30.0, 30.0])
    )

    # A custom reducer and repeated sharing operate on current projected values.
    mean = base.share(
        "g", groups=[0, 0, 1, 1], reduce=lambda values: values.mean(axis=0)
    )
    np.testing.assert_array_equal(mean._fields["g"].slots, np.array([15.0, 35.0]))
    regrouped = shared.share(
        "g", groups=[0, 1, 1, 2], reduce=lambda values: values.mean(axis=0)
    )
    np.testing.assert_array_equal(
        regrouped._fields["g"].slots, np.array([10.0, 20.0, 30.0])
    )

    # Unsharing creates one current slot per site and cannot recover discarded values.
    unshared = shared.unshare("g")
    np.testing.assert_array_equal(unshared._fields["g"].groups, np.arange(4))
    np.testing.assert_array_equal(
        unshared._fields["g"].slots, np.array([10.0, 10.0, 30.0, 30.0])
    )

    with pytest.raises(ValueError, match="unknown field"):
        base.share("missing")
    with pytest.raises(ValueError, match="unknown field"):
        base.unshare("missing")


def test_shared_fields_extend_new_and_overlapping_support():
    base = Model(Cable(4)).insert(Field("g", [10.0, 20.0], index=[0, 1]))
    shared = base.share("g")

    # Disjoint support added later is independent of the existing shared slot.
    extended = shared.insert(Field("g", [30.0, 40.0], index=[2, 3]))
    field = extended._fields["g"]
    np.testing.assert_array_equal(_project(field), np.array([10.0, 10.0, 30.0, 40.0]))
    assert field.groups[0] == field.groups[1]
    assert len(np.unique(field.groups)) == 3

    # Right-wins overlap replaces defaults and can split an existing shared group.
    overlap = shared.insert(Field("g", [50.0, 60.0], index=[1, 2]))
    np.testing.assert_array_equal(
        _project(overlap._fields["g"]), np.array([10.0, 50.0, 60.0])
    )
    assert len(np.unique(overlap._fields["g"].groups)) == 3


def test_same_name_mechanism_insertion_rebuilds_merged_support():
    base = Model(Cable(5)).insert(Leak(index=[1, 2])).share("Leak.g")
    original = base._fields["Leak.g"]

    # A same-name insertion overwrites declarations over the merged support.
    extended = base.insert(Leak(index=[2, 4]))
    field = extended._fields["Leak.g"]
    np.testing.assert_array_equal(extended._mechs["leak"].index, np.array([1, 2, 4]))
    np.testing.assert_array_equal(_project(field), np.array([0.3, 0.3, 0.3]))
    np.testing.assert_array_equal(field.groups, np.array([0, 1, 2]))

    # A redundant same-name insertion still overwrites the declaration.
    redundant = base.insert(Leak(index=[2]))
    assert redundant._fields["Leak.g"] is not original
    np.testing.assert_array_equal(redundant._fields["Leak.g"].index, np.array([1, 2]))


def test_declarations_use_identity_equality():
    # NumPy-backed declarations compare by identity without ambiguous array equality.
    assert Field("x", 1.0, index=[0]) != Field("x", 1.0, index=[0])
    assert Cable(1) != Cable(1)
    assert Na() != Na()
    assert Model(Cable(1)) != Model(Cable(1))


def test_late_state_fields_and_mechanism_placement():
    class ReadsState(Channel):
        states = {"x": None, "v": None}

    base = Model(Cable(3))
    referenced = base.insert(ReadsState(index=[1]))

    # None declarations are read dependencies and do not create or promote Fields.
    assert "x" not in referenced._fields
    model = referenced.insert(Field("x", 1.0, index=[1]))
    assert not model._fields["x"].dynamic

    # Dynamic placement is explicit when a read dependency is also integrated.
    dynamic = referenced.insert(Field("x", 1.0, dynamic=True, index=[1]))
    assert dynamic._fields["x"].dynamic

    # Mechanism sites must belong to the Model morphology.
    with pytest.raises(AssertionError, match="model compartments"):
        base.insert(Leak(index=[-1]))
    with pytest.raises(AssertionError, match="model compartments"):
        base.insert(Leak(index=[3]))
