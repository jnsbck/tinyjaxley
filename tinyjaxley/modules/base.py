from re import M
import equinox as eqx
from jax import Array
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Union

from ..mechanisms.mechanism import Mechanism
from ..mechanisms.channel import Channel
from ..mechanisms.external import Stimulus, Clamp
from ..utils import is_instance_of, tree_getter, stack_leaves
from jax import vmap

from typing import Optional
from operator import attrgetter


class Module(eqx.Module):
    l: Array = eqx.field(converter=jnp.array)
    r: Array = eqx.field(converter=jnp.array)
    c: Array = eqx.field(converter=jnp.array)
    ra: Array = eqx.field(converter=jnp.array)
    x: Array = eqx.field(converter=jnp.array)
    y: Array = eqx.field(converter=jnp.array)
    z: Array = eqx.field(converter=jnp.array)
    channels: dict[str, Channel]
    stimuli: dict[str, Stimulus]  # add to state or current
    clamps: dict[str, Clamp]  # set state or current
    # synapses: List[Synapse]

    parents: Array = eqx.field(converter=jnp.array)
    index: Array = eqx.field(converter=jnp.array)
    id: Array = eqx.field(converter=jnp.array)

    edges: Array = eqx.field(converter=jnp.array)

    def __init__(
        self,
        l: Array = 10.0,
        r: Array = 1.0,
        c: Array = 1.0,
        ra: Array = 5000.0,
        x: Array = 0.0,
        y: Array = 0.0,
        z: Array = 0.0,
        parents: Array = -1,
        index: Array = 0,
        id: Array = 0,
        key: str = None,
    ):
        self.l = l
        self.r = r
        self.c = c
        self.ra = ra
        self.x = x
        self.y = y
        self.z = z
        self.channels = {}
        self.stimuli = {}
        self.clamps = {}

        self.parents = parents
        self.index = index
        self.id = id

        # TODO: Add post init to initialize edges
        self.edges = self._edges_init()  # needs to be rerun if parents change

    @property
    def area(self):
        return 2.0 * jnp.pi * self.r * self.l  # um²

    @property
    def num_comps(self):
        return self.l.size

    def g_coupling(self, i, j):
        """
        from `https://en.wikipedia.org/wiki/Compartmental_neuron_models`.
        `rius`: um, `Ra`: ohm cm, `l`: um, `g`: mS / cm^2
        """
        r_i, l_i, ra_i = self.r[i], self.l[i], self.ra[i]
        r_j, l_j, ra_j = self.r[j], self.l[j], self.ra[j]
        g_ij = r_i * r_j**2 / (ra_i * r_j**2 * l_i + ra_j * r_i**2 * l_j) / l_i
        return g_ij * 1e7

    def G_sparse(self):
        i, j = self.edges.T
        n = self.num_comps
        g_ij = vmap(self.g_coupling)(i, j)

        # Concatenate off-diagonal (edges) and diagonal entries
        rows = jnp.concatenate([i, jnp.arange(n)])
        cols = jnp.concatenate([j, jnp.arange(n)])
        inds = jnp.stack([rows, cols], axis=1)
        values = jnp.concatenate([g_ij, -jnp.bincount(i, weights=g_ij, length=n)])

        return values, inds

    def dgates(self, t, u, v):
        du = jax.tree.map(
            lambda c: c(t, u.get(c.name, {}), u["v"][c.index]),
            self.channels,
            is_leaf=is_instance_of(Channel),
        )
        return du

    def compute_i_total(self, t, u, v):
        def sum_i(i0, c):
            u_ = u.get(c.name, {})
            i = c.i(t, u_, u["v"][c.index])
            return i0.at[c.index].add(i)

        i0 = jnp.zeros(self.num_comps)
        i_int = jax.tree.reduce(
            sum_i, self.channels, i0, is_leaf=is_instance_of(Channel)
        )
        i_ext = jax.tree.reduce(
            sum_i, self.stimuli, i0, is_leaf=is_instance_of(Stimulus)
        )
        # i_clamp = jax.tree.map(sum_i, self.clamps, is_leaf=is_instance(Clamp))

        i_ext_total = jax.tree.reduce(lambda x, y: x + y, i_ext, initializer=0.0)
        i_int_total = jax.tree.reduce(lambda x, y: x + y, i_int, initializer=0.0)

        return i_ext_total * 1e5 / self.area - i_int_total * 1e3

    def __call__(self, t, u, args=None):
        # TODO: Add clamping (for states and currents)
        # TODO: Add state sharing

        du = self.dgates(t, u, u["v"])

        i_total = self.compute_i_total(t, u, u["v"])
        dv_i = i_total / self.c

        i, j = self.edges.T
        dv_ij = vmap(self.g_coupling)(i, j) * (u["v"][j] - u["v"][i])
        du["v"] = dv_i + jnp.bincount(i, weights=dv_ij, length=len(dv_i))
        return du

    def _edges_init(self):
        edges = jnp.stack([self.parents, self.index], axis=1)
        edges = edges[edges[:, 0] != -1]
        edges = jnp.concatenate([edges, edges[:, ::-1]])
        edges = edges[jnp.lexsort((edges[:, 1], edges[:, 0]))]
        return edges

    # TODO: Add branchpoints
    # def _insert_branchpoints(self):
    #     pass

    def init(self, t, u=None):
        is_channel = lambda x: isinstance(x, Channel)
        v_at = lambda u, idx: u["v"][idx] if u["v"].size > 1 else u["v"]

        u = {} if u is None else u
        u["v"] = jnp.full(self.num_comps, -70.0) if "v" not in u else u["v"]
        u0 = jax.tree.map(
            lambda c: c.init(t, u[c.name] if c.name in u else {}, v_at(u, c.index)),
            self.channels,
            is_leaf=is_channel,
        )
        u0["v"] = u["v"]
        return u0

    def set(self, path_str: str, value: Array):
        return self.at[self.index].set(path_str, value)

    def insert(self, mech: Union[Channel, Stimulus, Clamp]):
        return self.at[self.index].insert(mech)

    # def remove(self, path: str):
    #     pass

    # def group(self, index)
    #     pass

    @property
    def at(self):
        return ModuleIndexer(self)

    def __getitem__(self, index):
        return self.at[index].get()

    def pandas(self):
        df = pd.DataFrame()
        cols = ["id", "l", "r", "c", "ra", "x", "y", "z"]

        data = []
        for k in cols:
            data.append(np.array(getattr(self, k)))

        for c in self.channels.values():
            for k in c.__annotations__.keys():
                cols.append(c.name + "." + k)
                col = np.full(self.num_comps, np.nan)
                col[c.index] = np.array(getattr(c, k))
                data.append(col)

        data = np.column_stack(data)
        df = pd.DataFrame(
            data, columns=cols, index=np.array(jnp.atleast_1d(self.index))
        )
        df["id"] = df["id"].astype(int)
        return df

    def graph(self):
        df = self.pandas()
        G = nx.Graph()
        G.add_edges_from(np.array(self.edges))
        nx.set_node_attributes(G, df.to_dict(orient="index"))
        return G

    def vis(self, dims=(0, 1), **kwargs):
        G = self.graph()
        xyz = jnp.stack([self.x, self.y, self.z], axis=1)[:, dims]
        nx.draw(G, pos=xyz, **kwargs)


class ModuleIndexer(eqx.Module):
    _module: Module
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, module: Module):
        self._module = module
        self.index = jnp.array([])

    def __repr__(self):
        return f"{self._module.__class__.__name__}@{self.index}"

    def __getitem__(self, index: Array):
        self_at = eqx.tree_at(lambda x: x.index, self, index)
        return self_at

    def _get_mech_index(self, path_str: str):
        getter = tree_getter(path_str)
        mech_idx = getter(self._module).index
        return jnp.intersect1d(mech_idx, self.index, return_indices=True)[1]

    def set(self, path_str: str, value: Array):
        path_str = path_str.lstrip(".")

        if path_str.startswith(("channels", "stimuli", "clamps")):
            mech_path, mech_attr_path = path_str.split(".", 1)
            mech_getter = tree_getter(mech_path)
            index = self._get_mech_index(mech_path)
            mech = mech_getter(self._module).at[index].set(mech_attr_path, value)
            return eqx.tree_at(mech_getter, self._module, mech)

        getter = tree_getter(path_str)
        replace_fn = lambda x: x.at[self.index].set(value)
        return eqx.tree_at(getter, self._module, replace_fn=replace_fn)

    def _get_mechs(self):
        def get_mechs(x):
            if isinstance(x, Mechanism):
                at = jnp.intersect1d(x.index, self.index, return_indices=True)[1]
                get_at = lambda x: x.at[at].get() if eqx.is_array(x) else x
                return jax.tree.map(get_at, x)
            return x.at[self.index].get()

        is_mech = lambda x: isinstance(x, Mechanism)
        return jax.tree.map(get_mechs, self._module, is_leaf=is_mech)

    def get(self, path_str: Optional[str] = None):
        if path_str is None:
            self_at = self._get_mechs()
            self_at = eqx.tree_at(lambda x: x.edges, self_at, self_at._edges_init())
            return self_at

        path_str = path_str.lstrip(".")
        if path_str.startswith(("channels", "stimuli", "clamps")):
            mech_path, mech_attr_path = path_str.split(".", 1)
            mech_getter = tree_getter(mech_path)
            index = self._get_mech_index(mech_path)
            return mech_getter(self._module).at[index].get(mech_attr_path)

        getter = tree_getter(path_str)
        replace_fn = lambda x: x.at[self.index].get()
        return eqx.tree_at(getter, self._module, replace_fn=replace_fn)

    def insert(self, mech: Union[Channel, Stimulus, Clamp]):
        path_str = "channels" if isinstance(mech, Channel) else None
        path_str = "stimuli" if isinstance(mech, Stimulus) else path_str
        path_str = "clamps" if isinstance(mech, Clamp) else path_str

        getter = attrgetter(path_str)
        existing_mechs = getter(self._module)
        at = self.index

        if self._module.index.ndim == 0:
            new_mech = mech
        elif mech.name in existing_mechs:
            old_mech = existing_mechs[mech.name]
            comb_inds = jnp.union1d(old_mech.index, at)
            new_mech = jax.tree.map(stack_leaves, *[mech] * comb_inds.size)
            new_mech = eqx.tree_at(lambda x: x.index, new_mech, comb_inds)

            where = jnp.intersect1d(old_mech.index, comb_inds, return_indices=True)[1]
            set_fn = lambda x, y: x.at[where].set(y) if eqx.is_array(x) else x
            new_mech = jax.tree.map(set_fn, new_mech, old_mech)
        else:
            new_mech = jax.tree.map(stack_leaves, *[mech] * at.size)
            new_mech = eqx.tree_at(lambda x: x.index, new_mech, at)

        all_mechs = {**existing_mechs, mech.name: new_mech}
        return eqx.tree_at(getter, self._module, all_mechs)
