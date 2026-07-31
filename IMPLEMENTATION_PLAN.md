# Implementation Plan v0.0.1 — Passive Cable First

## Goal

Get a passive cable working: Leak + Axial + CurrentClamp on 32 compartments.
Then add HH.  Keep the lowering simple and concrete.

---

## Package Structure

```
nex/
  __init__.py
  mechanisms/
    __init__.py
    mechanism.py        # Mechanism, Channel, EdgeMechanism
    channel.py          # Na, K, Leak (+ rate functions at module level)
    external.py         # CurrentClamp
    axial.py            # Axial
  model/
    __init__.py
    base.py             # Model builder
  utils/
    __init__.py
    fields.py           # Field, State, Param
    indexing.py          # Layout, compute_gather_idx
    kernels.py          # ProjectKernel, EdgeKernel
    lower.py            # lower_model → LoweredModel
  solve.py              # simulate
  optimize.py           # stubs
  io/
    __init__.py
  tests/
    test_fields.py
    test_indexing.py
    test_cable.py       # milestone 1: passive cable
    test_hh.py          # milestone 2: HH cable
```

---

## Step 1: `nex/utils/fields.py`

```python
import jax.numpy as jnp

class Field:
    """Base for owned fields.  State and Param are thin subclasses."""
    __slots__ = ("ref", "value", "train", "group", "local_name")

    def __init__(self, ref: str, value, *, train=False, group=None):
        assert isinstance(ref, str) and len(ref) > 0 and not ref.endswith(".")
        self.ref = ref
        self.value = jnp.asarray(value)
        self.train = train
        self.group = group
        self.local_name = ref.rsplit(".", 1)[-1]
        assert self.local_name.isidentifier(), f"'{self.local_name}' not valid"

    def replace_value(self, value):
        return type(self)(self.ref, value, train=self.train, group=self.group)

    def __repr__(self):
        return f"{type(self).__name__}({self.ref!r}, {self.value})"

class State(Field): pass
class Param(Field): pass
```

---

## Step 2: `nex/utils/indexing.py`

```python
import jax.numpy as jnp
from math import prod

class Layout:
    __slots__ = ("ref", "kind", "start", "shape", "n_storage", "group", "width", "size")

    def __init__(self, ref, kind, start, shape, n_storage, group):
        self.ref = ref
        self.kind = kind
        self.start = start
        self.shape = tuple(shape) if shape else ()
        self.n_storage = n_storage
        self.group = jnp.asarray(group, dtype=jnp.int32)
        self.width = prod(self.shape) if self.shape else 1
        self.size = n_storage * self.width

def compute_gather_idx(layout, comp_ids):
    comp_ids = jnp.asarray(comp_ids, dtype=jnp.int32)
    slot_ids = layout.group[comp_ids]
    if layout.width == 1:
        return (layout.start + slot_ids).astype(jnp.int32)
    offsets = jnp.arange(layout.width, dtype=jnp.int32).reshape(layout.shape)
    slot_exp = slot_ids.reshape(-1, *([1] * len(layout.shape)))
    return (layout.start + slot_exp * layout.width + offsets).astype(jnp.int32)
```

---

## Step 3: `nex/mechanisms/mechanism.py`

```python
import jax.numpy as jnp
from nex.utils.fields import Field, State, Param

def get_ref(entry):
    return entry.ref if isinstance(entry, Field) else entry

def get_local_name(entry):
    return get_ref(entry).rsplit(".", 1)[-1]

def is_owned(entry):
    return isinstance(entry, Field)

def validate_reads(reads, label=""):
    names = [get_local_name(e) for e in reads]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate local names in {label}: {names}")


class Mechanism:
    domain = "comp"
    states = ()
    params = ()
    current = None
    is_density = True

    def __init__(self, **overrides):
        self.states = tuple(self.__class__.states)
        self.params = tuple(self.__class__.params)
        if overrides:
            self.states = tuple(self._apply(e, overrides) for e in self.states)
            self.params = tuple(self._apply(e, overrides) for e in self.params)
            if overrides:
                raise ValueError(f"Unknown overrides: {set(overrides)}")
        validate_reads(self.states, f"{type(self).__name__}.states")
        validate_reads(self.params, f"{type(self).__name__}.params")

    @staticmethod
    def _apply(entry, overrides):
        if isinstance(entry, Field) and entry.local_name in overrides:
            return entry.replace_value(overrides.pop(entry.local_name))
        return entry

    def init(self, t, u, p, args): return None
    def vf(self, t, u, p, args):   return jnp.array([])
    def i(self, t, u, p, args):    return 0.0

class Channel(Mechanism):
    is_density = True

class EdgeMechanism(Mechanism):
    domain = "edge"
    is_density = False
    edge_states = ()
    edge_params = ()

    def i(self, t, u_pre, u_post, p_pre, p_post, p_edge, args):   return 0.0
    def vf(self, t, u_pre, u_post, p_pre, p_post, p_edge, args):  return jnp.array([])
```

---

## Step 4: `nex/mechanisms/channel.py`

Leak (sufficient for passive cable), then Na/K for milestone 2.

```python
import jax.numpy as jnp
from nex.mechanisms.mechanism import Channel
from nex.utils.fields import State, Param

# ── Rate functions (module-level, used by Na/K) ─────────────────

def alpha_m(v): return 0.1 * (v + 40) / (1 - jnp.exp(-(v + 40) / 10))
def beta_m(v):  return 4.0 * jnp.exp(-(v + 65) / 18)
def alpha_h(v): return 0.07 * jnp.exp(-(v + 65) / 20)
def beta_h(v):  return 1.0 / (1 + jnp.exp(-(v + 35) / 10))
def alpha_n(v): return 0.01 * (v + 55) / (1 - jnp.exp(-(v + 55) / 10))
def beta_n(v):  return 0.125 * jnp.exp(-(v + 65) / 80)


# ── Channels ────────────────────────────────────────────────────

class Leak(Channel):
    states = ("v",)
    params = (Param("leak.gbar", 0.3), Param("leak.e", -54.3))
    current = "ileak"

    def i(self, t, u, p, args):
        return -p.gbar * (u.v - p.e)


class Na(Channel):
    states = ("v", State("na.m", 0.05), State("na.h", 0.60))
    params = (Param("na.gbar", 120.0), Param("ions.ena", 50.0))
    current = "ina"

    def i(self, t, u, p, args):
        return -p.gbar * u.m**3 * u.h * (u.v - p.ena)

    def init(self, t, u, p, args):
        am, bm = alpha_m(u.v), beta_m(u.v)
        ah, bh = alpha_h(u.v), beta_h(u.v)
        return jnp.array([am / (am + bm), ah / (ah + bh)])

    def vf(self, t, u, p, args):
        dm = alpha_m(u.v) * (1 - u.m) - beta_m(u.v) * u.m
        dh = alpha_h(u.v) * (1 - u.h) - beta_h(u.v) * u.h
        return jnp.array([dm, dh])


class K(Channel):
    states = ("v", State("k.n", 0.32))
    params = (Param("k.gbar", 36.0), Param("ions.ek", -77.0))
    current = "ik"

    def i(self, t, u, p, args):
        return -p.gbar * u.n**4 * (u.v - p.ek)

    def init(self, t, u, p, args):
        an, bn = alpha_n(u.v), beta_n(u.v)
        return jnp.array([an / (an + bn)])

    def vf(self, t, u, p, args):
        return jnp.array([alpha_n(u.v) * (1 - u.n) - beta_n(u.v) * u.n])
```

---

## Step 5: `nex/mechanisms/external.py`

```python
import jax.numpy as jnp
from nex.mechanisms.mechanism import Mechanism
from nex.utils.fields import Param

class CurrentClamp(Mechanism):
    states = ()
    params = (Param("iclamp.amp", 0.1), Param("iclamp.t0", 10.0), Param("iclamp.t1", 20.0))
    current = "i_ext"
    is_density = False

    def i(self, t, u, p, args):
        return p.amp * ((t >= p.t0) & (t <= p.t1))
```

---

## Step 6: `nex/mechanisms/axial.py`

```python
import jax.numpy as jnp
from nex.mechanisms.mechanism import EdgeMechanism

class Axial(EdgeMechanism):
    states = ("v",)
    params = ("rad", "len", "res_ax")
    current = "i_axial"

    def i(self, t, u_pre, u_post, p_pre, p_post, p_edge, args):
        cross_pre = jnp.pi * p_pre.rad**2
        cross_post = jnp.pi * p_post.rad**2
        r_pre = p_pre.res_ax * p_pre.len / (2 * cross_pre)
        r_post = p_post.res_ax * p_post.len / (2 * cross_post)
        g = 1.0 / (r_pre + r_post)
        return g * (u_pre.v - u_post.v)
```

---

## Step 7: `nex/model/base.py`

The Model has `states` and `params` just like a mechanism — these are the
base fields for every compartment.  No special geometry.

```python
import jax.numpy as jnp
from nex.utils.fields import State, Param
from nex.mechanisms.mechanism import EdgeMechanism

class Model:
    def __init__(self, ncomp,
                 states=(State("v", -65.0),),
                 params=(Param("rad", 1.0), Param("len", 10.0),
                         Param("cap", 1.0), Param("res_ax", 100.0))):
        self.ncomp = ncomp
        self.states = states
        self.params = params
        self.insertions = []
        self.connections = []
        self.records = []

    @property
    def comps(self):
        return _Accessor(self)

    def connect(self, pre, post, mech):
        assert isinstance(mech, EdgeMechanism)
        self.connections.append((
            jnp.asarray(pre, dtype=jnp.int32),
            jnp.asarray(post, dtype=jnp.int32),
            mech,
        ))

    def lower(self):
        from nex.utils.lower import lower_model
        return lower_model(self)


class _Accessor:
    def __init__(self, model): self._m = model
    @property
    def at(self): return _Indexer(self._m)

class _Indexer:
    def __init__(self, model): self._m = model
    def __getitem__(self, idx):
        n = self._m.ncomp
        if isinstance(idx, int): ids = (idx,)
        elif isinstance(idx, slice): ids = tuple(range(*idx.indices(n)))
        else: ids = tuple(int(i) for i in idx)
        return _Selection(self._m, ids)

class _Selection:
    def __init__(self, model, ids): self._m, self._ids = model, ids
    def insert(self, mech, *, group=None):
        self._m.insertions.append((mech, self._ids, group or {}))
    def record(self, ref):
        self._m.records.append((ref, self._ids))
```

---

## Step 8: `nex/utils/kernels.py`

```python
import jax
import jax.numpy as jnp
from jax import vmap

class ProjectKernel:
    def __init__(self, mech, UCls, PCls,
                 u_gi, p_gi, scatter_flat, comp_ids,
                 aux_si, rad_idx, len_idx,
                 has_vf, has_i, has_init, is_density):
        self.mech = mech
        self.UCls, self.PCls = UCls, PCls
        self.u_gi, self.p_gi = u_gi, p_gi
        self.scatter_flat = scatter_flat
        self.comp_ids = comp_ids
        self.aux_si = aux_si
        self.rad_idx, self.len_idx = rad_idx, len_idx
        self.has_vf = has_vf
        self.has_i = has_i
        self.has_init = has_init
        self.is_density = is_density
        if has_vf:   self._vf   = vmap(mech.vf,   in_axes=(None, 0, 0, None))
        if has_i:    self._i    = vmap(mech.i,     in_axes=(None, 0, 0, None))
        if has_init: self._init = vmap(mech.init,  in_axes=(None, 0, 0, None))

    def __call__(self, t, u, p, args, du, aux, i_rhs):
        ul = self.UCls(*(u[i] for i in self.u_gi))
        pl = self.PCls(*(p[i] for i in self.p_gi))

        if self.has_i:
            iv = self._i(t, ul, pl, args)
            if not self.is_density:
                area = 2 * jnp.pi * p[self.rad_idx] * p[self.len_idx]
                iv = iv / area
            i_rhs = i_rhs.at[self.comp_ids].add(iv)
            if self.aux_si is not None:
                aux = aux.at[self.aux_si].add(iv)

        if self.has_vf:
            dv = self._vf(t, ul, pl, args)
            du = du.at[self.scatter_flat].add(dv.ravel())

        return du, aux, i_rhs

    def run_init(self, t, u, p, args):
        if not self.has_init: return u
        ul = self.UCls(*(u[i] for i in self.u_gi))
        pl = self.PCls(*(p[i] for i in self.p_gi))
        vals = self._init(t, ul, pl, args)
        return u.at[self.scatter_flat].set(vals.ravel())


class EdgeKernel:
    def __init__(self, mech, UCls, PCls, PEdgeCls,
                 u_pre_gi, u_post_gi, p_pre_gi, p_post_gi, p_edge_gi,
                 post_comp_ids, aux_si, rad_post_idx, len_post_idx,
                 scatter_flat, has_i, has_vf, is_density):
        self.mech = mech
        self.UCls, self.PCls, self.PEdgeCls = UCls, PCls, PEdgeCls
        self.u_pre_gi, self.u_post_gi = u_pre_gi, u_post_gi
        self.p_pre_gi, self.p_post_gi = p_pre_gi, p_post_gi
        self.p_edge_gi = p_edge_gi
        self.post_ids = post_comp_ids
        self.aux_si = aux_si
        self.rad_pi, self.len_pi = rad_post_idx, len_post_idx
        self.scatter_flat = scatter_flat
        self.has_i, self.has_vf = has_i, has_vf
        self.is_density = is_density
        ax = (None, 0, 0, 0, 0, 0, None)
        if has_i:  self._i  = vmap(mech.i,  in_axes=ax)
        if has_vf: self._vf = vmap(mech.vf, in_axes=ax)

    def __call__(self, t, u, p, args, du, aux, i_rhs):
        u_pre  = self.UCls(*(u[i] for i in self.u_pre_gi))
        u_post = self.UCls(*(u[i] for i in self.u_post_gi))
        p_pre  = self.PCls(*(p[i] for i in self.p_pre_gi))
        p_post = self.PCls(*(p[i] for i in self.p_post_gi))
        p_edge = (self.PEdgeCls(*(p[i] for i in self.p_edge_gi))
                  if self.p_edge_gi else self.PEdgeCls())

        if self.has_i:
            iv = self._i(t, u_pre, u_post, p_pre, p_post, p_edge, args)
            if not self.is_density:
                area = 2 * jnp.pi * p[self.rad_pi] * p[self.len_pi]
                iv = iv / area
            i_rhs = i_rhs.at[self.post_ids].add(iv)
            if self.aux_si is not None:
                aux = aux.at[self.aux_si].add(iv)

        if self.has_vf:
            dv = self._vf(t, u_pre, u_post, p_pre, p_post, p_edge, args)
            du = du.at[self.scatter_flat].add(dv.ravel())

        return du, aux, i_rhs
```

---

## Step 9: `nex/utils/lower.py`

This is the critical step.  I'm writing it out fully because it must be
concrete and simple.

```python
import jax.numpy as jnp
from collections import namedtuple
from nex.utils.fields import Field, State, Param
from nex.utils.indexing import Layout, compute_gather_idx
from nex.utils.kernels import ProjectKernel, EdgeKernel
from nex.mechanisms.mechanism import (
    Mechanism, EdgeMechanism, get_ref, get_local_name, is_owned
)


class RecordPlan:
    __slots__ = ("ref", "source", "gather_idx")
    def __init__(self, ref, source, gather_idx):
        self.ref, self.source, self.gather_idx = ref, source, gather_idx


def lower_model(model):
    """
    Convert Model → LoweredModel.

    The algorithm:
      1. Collect all unique fields (from model + mechanisms + edges).
      2. Resolve groups, allocate flat regions.
      3. Build u0 and p0 flat vectors.
      4. Build kernels with precomputed gather/scatter indices.
      5. Build record plans.
    """
    N = model.ncomp
    all_comps = jnp.arange(N, dtype=jnp.int32)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Collect all fields and resolve groups
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Registry: ref → {kind, group, value, shape}
    # kind: "state" or "param"
    # group: jnp array [N] of slot ids (compacted later)
    # value: default value
    state_reg = {}  # ref → {"group": [N], "value": array, "shape": tuple}
    param_reg = {}

    def register_field(field, comp_ids, group_overrides, registry):
        """Register an owned field.  Merges if already exists (last wins)."""
        ref = field.ref
        comp_ids = jnp.asarray(comp_ids, dtype=jnp.int32)

        # Resolve group for these compartments
        if ref in group_overrides:
            g_val = group_overrides[ref]
            if isinstance(g_val, int):
                g = jnp.full(len(comp_ids), g_val, dtype=jnp.int32)
            else:
                g = jnp.asarray(g_val, dtype=jnp.int32)
        elif field.group is not None:
            g_val = field.group
            if isinstance(g_val, int):
                g = jnp.full(len(comp_ids), g_val, dtype=jnp.int32)
            else:
                g = jnp.asarray(g_val, dtype=jnp.int32)
        else:
            # Default: per-comp
            g = comp_ids

        if ref not in registry:
            group = jnp.full(N, -1, dtype=jnp.int32)
            group = group.at[comp_ids].set(g)
            registry[ref] = {
                "group": group,
                "value": field.value,
                "shape": field.value.shape,
            }
        else:
            # Merge: last wins per compartment
            entry = registry[ref]
            entry["group"] = entry["group"].at[comp_ids].set(g)
            entry["value"] = field.value

    # Register model's own fields (for all compartments)
    for entry in model.states:
        if is_owned(entry):
            register_field(entry, list(range(N)), {}, state_reg)

    for entry in model.params:
        if is_owned(entry):
            register_field(entry, list(range(N)), {}, param_reg)

    # Register fields from comp mechanism insertions
    for mech, comp_ids, group_ov in model.insertions:
        for entry in mech.states:
            if is_owned(entry):
                register_field(entry, comp_ids, group_ov, state_reg)
        for entry in mech.params:
            if is_owned(entry):
                register_field(entry, comp_ids, group_ov, param_reg)

    # Register fields from edge connections
    # Edge mechs read compartment fields (bare strings) — no registration needed.
    # Edge-owned fields need their own allocation (per-edge, not per-comp).
    # For v0.0.1, edge_states and edge_params are empty, so skip.

    # Collect unique current names for aux
    aux_names = []
    for mech, comp_ids, _ in model.insertions:
        if mech.current and mech.current not in aux_names:
            aux_names.append(mech.current)
    for _, _, edge_mech in model.connections:
        if edge_mech.current and edge_mech.current not in aux_names:
            aux_names.append(edge_mech.current)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Compact groups and allocate flat regions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    state_layouts = {}
    offset = 0
    for ref, info in state_reg.items():
        group, n_storage = _compact(info["group"])
        shape = info["shape"]
        layout = Layout(ref, "state", offset, shape, n_storage, group)
        state_layouts[ref] = layout
        offset += layout.size
    state_size = offset

    param_layouts = {}
    offset = 0
    for ref, info in param_reg.items():
        group, n_storage = _compact(info["group"])
        shape = info["shape"]
        layout = Layout(ref, "param", offset, shape, n_storage, group)
        param_layouts[ref] = layout
        offset += layout.size
    param_size = offset

    aux_layouts = {}
    offset = 0
    for name in aux_names:
        group = jnp.arange(N, dtype=jnp.int32)  # always per-comp
        layout = Layout(name, "aux", offset, (), N, group)
        aux_layouts[name] = layout
        offset += layout.size
    aux_size = offset

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Build u0 and p0
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    u0 = jnp.zeros(state_size)
    for ref, info in state_reg.items():
        layout = state_layouts[ref]
        val = jnp.broadcast_to(info["value"], (layout.n_storage, *layout.shape))
        u0 = u0.at[layout.start : layout.start + layout.size].set(val.ravel())

    p0 = jnp.zeros(param_size)
    for ref, info in param_reg.items():
        layout = param_layouts[ref]
        val = jnp.broadcast_to(info["value"], (layout.n_storage, *layout.shape))
        p0 = p0.at[layout.start : layout.start + layout.size].set(val.ravel())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Build comp kernels
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    comp_kernels = []
    for mech, comp_ids, group_ov in model.insertions:
        comp_kernels.append(_build_comp_kernel(
            mech, comp_ids, state_layouts, param_layouts, aux_layouts, N))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. Build edge kernels
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    edge_kernels = []
    for pre_ids, post_ids, edge_mech in model.connections:
        edge_kernels.append(_build_edge_kernel(
            edge_mech, pre_ids, post_ids,
            state_layouts, param_layouts, aux_layouts, N))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. Record plans
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    record_plans = []
    for ref, comp_ids in model.records:
        comp_arr = jnp.asarray(comp_ids, dtype=jnp.int32)
        if ref in state_layouts:
            idx = compute_gather_idx(state_layouts[ref], comp_arr)
            record_plans.append(RecordPlan(ref, "state", idx))
        elif ref in aux_layouts:
            idx = compute_gather_idx(aux_layouts[ref], comp_arr)
            record_plans.append(RecordPlan(ref, "aux", idx))
        else:
            raise ValueError(f"Unknown record ref: {ref}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. Assemble
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    v_idx = compute_gather_idx(state_layouts["v"], all_comps)
    cap_idx = compute_gather_idx(param_layouts["cap"], all_comps)

    return LoweredModel(
        comp_kernels=tuple(comp_kernels),
        edge_kernels=tuple(edge_kernels),
        ncomp=N, state_size=state_size, param_size=param_size, aux_size=aux_size,
        v_idx=v_idx, cap_idx=cap_idx,
        u0=u0, p0=p0,
        record_plans=tuple(record_plans),
        state_layouts=state_layouts,
        param_layouts=param_layouts,
        aux_layouts=aux_layouts,
    )


# ── Helpers ──────────────────────────────────────────────────────

def _compact(group):
    """Remap used group ids to contiguous 0..n-1.  -1 stays -1."""
    used = group[group >= 0]
    if len(used) == 0:
        return group, 0
    unique = jnp.unique(used)
    mapping = {int(old): new for new, old in enumerate(unique)}
    new = jnp.array([mapping.get(int(g), -1) for g in group], dtype=jnp.int32)
    return new, len(unique)


def _make_namedtuple(name, reads):
    """Build a namedtuple class from a reads tuple."""
    local_names = [get_local_name(e) for e in reads]
    return namedtuple(name, local_names) if local_names else namedtuple(name, [])


def _gather_idxs(reads, layouts, comp_ids):
    """Build list of gather index arrays, one per entry in reads."""
    comp_ids = jnp.asarray(comp_ids, dtype=jnp.int32)
    idxs = []
    for entry in reads:
        ref = get_ref(entry)
        layout = layouts.get(ref)
        if layout is None:
            raise ValueError(f"Ref '{ref}' not found in layouts")
        idxs.append(compute_gather_idx(layout, comp_ids))
    return idxs


def _scatter_flat(reads, layouts, comp_ids):
    """Build flattened scatter index for owned fields only."""
    comp_ids = jnp.asarray(comp_ids, dtype=jnp.int32)
    parts = []
    for entry in reads:
        if is_owned(entry):
            idx = compute_gather_idx(layouts[entry.ref], comp_ids)
            parts.append(idx.ravel())
    if parts:
        return jnp.concatenate(parts)
    return jnp.array([], dtype=jnp.int32)


def _build_comp_kernel(mech, comp_ids, state_layouts, param_layouts, aux_layouts, N):
    comp_arr = jnp.asarray(comp_ids, dtype=jnp.int32)
    all_layouts = {**state_layouts, **param_layouts}

    UCls = _make_namedtuple(f"U_{type(mech).__name__}", mech.states)
    PCls = _make_namedtuple(f"P_{type(mech).__name__}", mech.params)

    u_gi = _gather_idxs(mech.states, state_layouts, comp_ids)
    p_gi = _gather_idxs(mech.params, param_layouts, comp_ids)
    sf   = _scatter_flat(mech.states, state_layouts, comp_ids)

    # Aux scatter
    aux_si = None
    if mech.current and mech.current in aux_layouts:
        aux_si = compute_gather_idx(aux_layouts[mech.current], comp_arr)

    # Area gather for point currents
    rad_idx = len_idx = None
    if not mech.is_density and "rad" in param_layouts and "len" in param_layouts:
        rad_idx = compute_gather_idx(param_layouts["rad"], comp_arr)
        len_idx = compute_gather_idx(param_layouts["len"], comp_arr)

    has_vf = type(mech).vf is not Mechanism.vf
    has_i = mech.current is not None
    has_init = type(mech).init is not Mechanism.init

    return ProjectKernel(
        mech, UCls, PCls, u_gi, p_gi, sf, comp_arr,
        aux_si, rad_idx, len_idx, has_vf, has_i, has_init, mech.is_density)


def _build_edge_kernel(mech, pre_ids, post_ids,
                       state_layouts, param_layouts, aux_layouts, N):
    pre_arr = jnp.asarray(pre_ids, dtype=jnp.int32)
    post_arr = jnp.asarray(post_ids, dtype=jnp.int32)

    UCls = _make_namedtuple(f"U_{type(mech).__name__}", mech.states)
    PCls = _make_namedtuple(f"P_{type(mech).__name__}", mech.params)
    PEdgeCls = _make_namedtuple(f"PE_{type(mech).__name__}", mech.edge_params)

    # Same fields, gathered from different compartments
    u_pre_gi  = _gather_idxs(mech.states, state_layouts, pre_ids)
    u_post_gi = _gather_idxs(mech.states, state_layouts, post_ids)
    p_pre_gi  = _gather_idxs(mech.params, param_layouts, pre_ids)
    p_post_gi = _gather_idxs(mech.params, param_layouts, post_ids)
    p_edge_gi = []  # edge-owned params (empty for v0.0.1)

    sf = jnp.array([], dtype=jnp.int32)  # no edge-owned states for v0.0.1

    aux_si = None
    if mech.current and mech.current in aux_layouts:
        aux_si = compute_gather_idx(aux_layouts[mech.current], post_arr)

    rad_pi = len_pi = None
    if not mech.is_density and "rad" in param_layouts and "len" in param_layouts:
        rad_pi = compute_gather_idx(param_layouts["rad"], post_arr)
        len_pi = compute_gather_idx(param_layouts["len"], post_arr)

    has_i = mech.current is not None
    has_vf = type(mech).vf is not EdgeMechanism.vf

    return EdgeKernel(
        mech, UCls, PCls, PEdgeCls,
        u_pre_gi, u_post_gi, p_pre_gi, p_post_gi, p_edge_gi,
        post_arr, aux_si, rad_pi, len_pi, sf,
        has_i, has_vf, mech.is_density)


# ── LoweredModel ────────────────────────────────────────────────

class LoweredModel:
    def __init__(self, comp_kernels, edge_kernels,
                 ncomp, state_size, param_size, aux_size,
                 v_idx, cap_idx, u0, p0, record_plans,
                 state_layouts, param_layouts, aux_layouts):
        self.comp_kernels = comp_kernels
        self.edge_kernels = edge_kernels
        self.ncomp = ncomp
        self.state_size = state_size
        self.param_size = param_size
        self.aux_size = aux_size
        self.v_idx = v_idx
        self.cap_idx = cap_idx
        self.u0 = u0
        self.p0 = p0
        self.record_plans = record_plans
        self.state_layouts = state_layouts
        self.param_layouts = param_layouts
        self.aux_layouts = aux_layouts

    def rhs(self, t, u, p, args=()):
        du = jnp.zeros(self.state_size)
        aux = jnp.zeros(self.aux_size)
        i_rhs = jnp.zeros(self.ncomp)

        for k in self.comp_kernels:
            du, aux, i_rhs = k(t, u, p, args, du, aux, i_rhs)
        for k in self.edge_kernels:
            du, aux, i_rhs = k(t, u, p, args, du, aux, i_rhs)

        dv = i_rhs / p[self.cap_idx]
        du = du.at[self.v_idx].add(dv)
        return du, aux

    def init(self, t=0.0, u=None, p=None, args=()):
        if u is None: u = self.u0
        if p is None: p = self.p0
        for k in self.comp_kernels:
            u = k.run_init(t, u, p, args)
        return u

    def record(self, u, aux):
        out = {}
        for rp in self.record_plans:
            src = u if rp.source == "state" else aux
            out[rp.ref] = src[rp.gather_idx]
        return out
```

---

## Step 10: `nex/solve.py`

```python
import jax
import jax.numpy as jnp

def simulate(model, u0, p, ts, args=()):
    dt = ts[1] - ts[0]
    def step(u, t):
        du, aux = model.rhs(t, u, p, args)
        u_next = u + dt * du
        rec = model.record(u_next, aux)
        return u_next, rec
    u_final, recs = jax.lax.scan(step, u0, ts[:-1])
    return {"ts": ts, "recs": recs, "u_final": u_final}
```

---

## Step 11: Tests

### `tests/test_cable.py` — Milestone 1

Passive cable: Leak + Axial + CurrentClamp.  No gating dynamics.
The voltage should spread from the stimulated compartment.

```python
import jax
import jax.numpy as jnp
from nex.model.base import Model
from nex.mechanisms.channel import Leak
from nex.mechanisms.external import CurrentClamp
from nex.mechanisms.axial import Axial
from nex.solve import simulate

def test_passive_cable():
    N = 32
    model = Model(ncomp=N)
    model.comps.at[:].insert(Leak())
    model.comps.at[0].insert(CurrentClamp(amp=0.1, t0=5.0, t1=40.0))

    pre = jnp.arange(N - 1)
    post = jnp.arange(1, N)
    model.connect(pre, post, Axial())
    model.connect(post, pre, Axial())

    model.comps.at[0].record("v")
    model.comps.at[N - 1].record("v")

    lowered = model.lower()
    u0 = lowered.u0  # no init needed for passive cable
    p = lowered.p0

    # Verify shapes
    print(f"state_size={lowered.state_size}, param_size={lowered.param_size}")
    print(f"aux_size={lowered.aux_size}")
    print(f"comp_kernels={len(lowered.comp_kernels)}")
    print(f"edge_kernels={len(lowered.edge_kernels)}")

    # Single RHS call
    du, aux = lowered.rhs(0.0, u0, p)
    print(f"du shape: {du.shape}, aux shape: {aux.shape}")

    # JIT
    rhs_jit = jax.jit(lowered.rhs)
    du2, aux2 = rhs_jit(0.0, u0, p)
    assert jnp.allclose(du, du2), "JIT mismatch"

    # Simulate
    ts = jnp.linspace(0, 50, 5001)
    sol = jax.jit(simulate, static_argnums=(0,))(lowered, u0, p, ts)

    v0 = sol["recs"]["v"]  # [5000, ...] voltage at comp 0
    # Stimulus should depolarize comp 0
    assert v0[:, 0].max() > -60.0, "Comp 0 should depolarize"
    # Signal should reach comp 31 but be attenuated
    print(f"V0 peak: {v0[:, 0].max():.1f}, V31 peak: {v0[:, 1].max():.1f}")

    print("Passive cable test passed!")
```

### `tests/test_hh.py` — Milestone 2

Add Na/K, verify spikes.

```python
def test_hh_cable():
    N = 32
    model = Model(ncomp=N)
    model.comps.at[:].insert(Na())
    model.comps.at[:].insert(K())
    model.comps.at[:].insert(Leak())
    model.comps.at[0].insert(CurrentClamp(amp=0.1, t0=10.0, t1=20.0))

    pre = jnp.arange(N - 1)
    post = jnp.arange(1, N)
    model.connect(pre, post, Axial())
    model.connect(post, pre, Axial())

    model.comps.at[0].record("v")
    model.comps.at[N - 1].record("v")

    lowered = model.lower()
    u0 = lowered.init()

    ts = jnp.linspace(0, 50, 5001)
    sol = jax.jit(simulate, static_argnums=(0,))(lowered, u0, lowered.p0, ts)

    v = sol["recs"]["v"]
    assert v[:, 0].max() > 0, "Comp 0 should spike"
    print("HH cable test passed!")
```

---

## Implementation Order

| # | File | Test |
|---|------|------|
| 1 | `utils/fields.py` | `test_fields.py` |
| 2 | `utils/indexing.py` | `test_indexing.py` |
| 3 | `mechanisms/mechanism.py` | inline |
| 4 | `mechanisms/channel.py` (Leak only first) | inline |
| 5 | `mechanisms/external.py` | inline |
| 6 | `mechanisms/axial.py` | inline |
| 7 | `model/base.py` | inline |
| 8 | `utils/kernels.py` | inline |
| 9 | `utils/lower.py` | **milestone 1: `test_cable.py`** |
| 10 | `solve.py` | **milestone 1** |
| 11 | Add Na/K to `channel.py` | **milestone 2: `test_hh.py`** |

---

## Roadmap

### v0.0.1 (this plan)
- flat u, p, aux with Layout + groups
- Leak, Na, K, CurrentClamp, Axial
- ProjectKernel, EdgeKernel
- gather → vmap → scatter
- forward Euler with lax.scan
- basic recording
- passive cable test + HH cable test

### v0.0.2
- voltage clamp (writes/phase/apply)
- calcium mechanisms (ICa, concentration, KCa)
- Diffrax integration
- refined init (steady-state, burn-in)
- SaveAt-like recording
- test against Jaxley (single comp, simple branch)

### v0.0.3
- branched cable models, branch points
- custom semi-implicit solver
- build → run → train workflow with optax
- API: remove, set, add, group editing
- introspection: __str__, vis(), df, nx, xyzr, loc
- sparse trainability definitions

### v0.0.4
- map/filter/edit params and states
- regression test suite (compile time, runtime, memory)
- SWC morphology support (reader, Model.from_swc, visualization)
- geometry-aware concentration dynamics (volume)
- synapses: current-based, conductance-based, plasticity
- recurrent networks, population connectivity
- performance: batched models, GPU benchmarks

### Future (v0.1+)
- ion tracking, Nernst equation
- stochastic channels
- gap junctions, extracellular potentials
- NEURON .mod import, NeuroML/NWB export
- distributed simulation (pmap)