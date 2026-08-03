import numpy as np

from tinycable import Field


def test_field_storage():
    scalar = Field("voltage", -65.0, dynamic=True)
    vector_default = np.array([0.1, 0.2], dtype=np.float32)
    vector = Field("conductance", vector_default)

    # Scalar and vector fields use the same leading slot axis.
    assert scalar.dynamic
    assert scalar.payload == ()
    assert scalar.alloc().shape == (1,)
    assert vector.payload == (2,)
    assert vector.alloc().shape == (1, 2)
    assert vector.alloc().dtype == np.float32

    # A declaration owns an immutable copy of its default.
    vector_default[0] = 9.0
    np.testing.assert_array_equal(vector.default, np.array([0.1, 0.2], np.float32))
    assert not vector.default.flags.writeable
