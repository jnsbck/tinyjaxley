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
from ..utils import tree_set_with_path, tree_apply_with_path
from jax import vmap

from typing import Optional


class Module(eqx.Module):
    l: Array = eqx.field(converter=jnp.array)
    r: Array = eqx.field(converter=jnp.array)
    c: Array = eqx.field(converter=jnp.array)
    ra: Array = eqx.field(converter=jnp.array)
    xyz: Array = eqx.field(converter=jnp.array)
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
        xyz: Array = jnp.array([0.0, 0.0, 0.0]),
        parents: Array = jnp.array(-1),
        index: Array = jnp.array(0),
        id: Array = jnp.array(0),
        key: str = None,
    ):
        self.l = l
        self.r = r
        self.c = c
        self.ra = ra
        self.channels = {}
        self.stimuli = {}
        self.clamps = {}
        self.xyz = xyz

        self.parents = parents
        self.index = index
        self.id = id

        # TODO: Add post init to initialize edges
        self.edges = self._edges_init()  # needs to be rerun if parents change

    @property
    def area(self):
        return 2.0 * jnp.pi * self.r * self.l  # um²

    def G(self, i, j):
        """
        from `https://en.wikipedia.org/wiki/Compartmental_neuron_models`.
        `rius`: um, `Ra`: ohm cm, `l`: um, `g`: mS / cm^2
        """
        r_i, l_i, ra_i = self.r[i], self.l[i], self.ra[i]
        r_j, l_j, ra_j = self.r[j], self.l[j], self.ra[j]
        g_ij = r_i * r_j**2 / (ra_i * r_j**2 * l_i + ra_j * r_i**2 * l_j) / l_i
        return g_ij * 1e7

    def __call__(self, t, u, args=None):
        is_instance = lambda cls: lambda x: isinstance(x, cls)
        # TODO: Add clamping (for states and currents)
        # TODO: Add state sharing
        i0 = jnp.zeros(self.l.size)

        def i_dist(c):
            u_ = u.get(c.name, {})
            i = c.i(t, u_, u["v"][c.index])
            return i0.at[c.index].set(i)

        i_int = jax.tree.map(i_dist, self.channels, is_leaf=is_instance(Channel))
        i_ext = jax.tree.map(i_dist, self.stimuli, is_leaf=is_instance(Stimulus))
        # i_clamp = jax.tree.map(i_dist, self.clamps, is_leaf=is_instance(Clamp))

        i_ext_total = jax.tree.reduce(lambda x, y: x + y, i_ext, initializer=0.0)
        i_int_total = jax.tree.reduce(lambda x, y: x + y, i_int, initializer=0.0)

        du = jax.tree.map(
            lambda c: c(t, u.get(c.name, {}), u["v"][c.index]),
            self.channels,
            is_leaf=is_instance(Channel),
        )

        dv_ii = (i_ext_total * 1e5 / self.area - i_int_total * 1e3) / self.c

        i, j = self.edges.T
        dv_ij = vmap(self.G)(i, j) * (u["v"][j] - u["v"][i])
        du["v"] = dv_ii + jnp.bincount(i, weights=dv_ij, length=len(dv_ii))

        return du

    def _edges_init(self):
        edges = jnp.stack([self.parents, self.index], axis=1)
        edges = edges[edges[:, 0] != -1]
        edges = jnp.concatenate([edges, edges[:, ::-1]])
        return edges

    # TODO: Add branchpoints
    # def _insert_branchpoints(self):
    #     pass

    def init(self, t, u=None):
        is_channel = lambda x: isinstance(x, Channel)
        v_at = lambda u, idx: u["v"][idx] if u["v"].size > 1 else u["v"]

        u = {} if u is None else u
        u["v"] = jnp.full(self.l.size, -70.0) if "v" not in u else u["v"]
        u0 = jax.tree.map(
            lambda c: c.init(t, u[c.name] if c.name in u else {}, v_at(u, c.index)),
            self.channels,
            is_leaf=is_channel,
        )
        u0["v"] = u["v"]
        return u0

    def set(self, set_dict):
        return tree_set_with_path(self, set_dict)

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
        cols = ["id", "l", "r", "c", "ra"]

        data = []
        for k in cols:
            data.append(np.array(getattr(self, k)))

        for c in self.channels.values():
            for k in c.__annotations__.keys():
                cols.append(c.name + "." + k)
                col = np.full(self.l.size, np.nan)
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
        xyz = self.xyz[:, dims]
        nx.draw(G, pos=xyz, **kwargs)


class ModuleIndexer(eqx.Module):
    _module: Module
    index: Array = eqx.field(converter=jnp.array)

    def __init__(self, module: Module):
        self._module = module
        self.index = module.index

    def __repr__(self):
        return f"{self._module.__class__.__name__}@{self.index}"

    def __getitem__(self, index: Array):
        self_at = eqx.tree_at(lambda x: x.index, self, index)
        return self_at

    def set(self, set_dict: dict):
        m = self._module
        self_at = self.get()
        self_at = tree_set_with_path(self_at, set_dict)
        # merge m and self_at together TODO: write merge util
        # return updated_m

    def insert(self, mech: Union[Channel, Stimulus, Clamp]):
        at = jnp.atleast_1d(self.index)
        mech = jax.tree.map(
            lambda *leaves: jnp.stack(leaves) if eqx.is_array(leaves[0]) else leaves[0],
            *[mech] * len(at),
        )
        mech = eqx.tree_at(lambda x: x.index, mech, at)

        m = self._module
        if isinstance(mech, Channel):
            channels = m.channels.copy()
            channels.update({mech.name: mech})
            return eqx.tree_at(lambda x: x.channels, m, channels)
        elif isinstance(mech, Stimulus):
            stimuli = m.stimuli.copy()
            stimuli.update({mech.name: mech})
            return eqx.tree_at(lambda x: x.stimuli, m, stimuli)
        elif isinstance(mech, Clamp):
            clamps = m.clamps.copy()
            clamps.update({mech.name: mech})
            return eqx.tree_at(lambda x: x.clamps, m, clamps)
        else:
            raise ValueError(f"Invalid type: {type(mech)}")

    def get(self):
        # TODO: Make jit-able
        index = jnp.atleast_1d(self.index)
        m = self._module

        # Filter compartment attributes
        filter_comp = (
            lambda x: x[index] if isinstance(x, Array) and x.size == m.l.size else x
        )
        self_at = jax.tree.map(filter_comp, m, is_leaf=lambda x: isinstance(x, Array))
        self_at = eqx.tree_at(lambda x: x.xyz, self_at, m.xyz[index])

        # Filter and remap edges
        self_at = eqx.tree_at(lambda x: x.edges, self_at, self_at._edges_init())

        # Filter mechanisms
        is_mech = lambda x: isinstance(x, Mechanism)
        get_mech = lambda x: x.at[index].get()
        channels = jax.tree.map(get_mech, self_at.channels, is_leaf=is_mech)
        stimuli = jax.tree.map(get_mech, self_at.stimuli, is_leaf=is_mech)
        clamps = jax.tree.map(get_mech, self_at.clamps, is_leaf=is_mech)
        self_at = eqx.tree_at(lambda x: x.channels, self_at, channels)
        self_at = eqx.tree_at(lambda x: x.stimuli, self_at, stimuli)
        self_at = eqx.tree_at(lambda x: x.clamps, self_at, clamps)

        return self_at
