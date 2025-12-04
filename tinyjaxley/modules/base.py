import equinox as eqx
from jax import Array
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import networkx as nx

from typing import List, Union
from ..mechanisms.channel import Channel
from ..mechanisms.external import Stimulus, Clamp
from ..utils import tree_set_with_path, tree_apply_with_path


class Module(eqx.Module):
    l: Array = eqx.field(converter=jnp.array)
    r: Array = eqx.field(converter=jnp.array)
    c: Array = eqx.field(converter=jnp.array)
    Ra: Array = eqx.field(converter=jnp.array)
    xyz: Array = eqx.field(converter=jnp.array)
    channels: dict[str, Channel]
    stimuli: dict[str, Stimulus]  # add to state or current
    clamps: dict[str, Clamp]  # set state or current
    # synapses: List[Synapse]
    parent: Array = eqx.field(converter=jnp.array)
    edges: Array = eqx.field(converter=jnp.array)

    def __init__(
        self, l: Array = 10.0, r: Array = 1.0, c: Array = 1.0, Ra: Array = 10_000.0
    ):
        self.l = l
        self.r = r
        self.c = c
        self.Ra = Ra
        self.channels = {}
        self.stimuli = {}
        self.clamps = {}
        self.parent = jnp.array(-1)
        self.xyz = jnp.array([0.0, 0.0, 0.0])
        self.edges = self._edges_init()  # needs to be rerun if parents change

    @property
    def area(self):
        return 2.0 * jnp.pi * self.r * self.l * 1e-4  # um² -> cm²

    def _edges_init(self):
        if self.parent.size > 1:
            inds = jnp.arange(self.parent.size)
            edges = jnp.stack([self.parent, inds], axis=1)
            edges = edges[edges[:, 0] != -1]
            return edges
        else:
            return jnp.array([]).reshape(0, 2)

    def G(self, i, j):
        """
        from `https://en.wikipedia.org/wiki/Compartmental_neuron_models`.
        `radius`: um, `Ra`: ohm cm, `l`: um, `g`: mS / cm^2
        """
        r_i, r_j = self.r[i], self.r[j]
        Ra_i, Ra_j = self.Ra[i], self.Ra[j]
        l_i, l_j = self.l[i], self.l[j]
        g = r_i * r_j**2 / (Ra_i * r_j**2 * l_i + Ra_j * r_i**2 * l_j) / l_i
        # TODO: check units
        return g * 1e7  # S/cm/um -> mS / cm²

    def __call__(self, t, u, args=None):
        # TODO: Use seperate diffrax.ODETerms for channels and comp??
        is_instance = lambda cls: lambda x: isinstance(x, cls)
        v_at = lambda u, c: u["v"][c.index] if u["v"].size > 1 else u["v"]

        # TODO: Add clamping

        i0 = jnp.zeros(self.l.size)

        def i_dist(c):
            v = v_at(u, c)
            u_ = u[c.name] if c.name in u else {}
            i = c.i(t, u_, v)
            return i0.at[c.index].set(i)

        i_int = jax.tree.map(i_dist, self.channels, is_leaf=is_instance(Channel))
        i_ext = jax.tree.map(i_dist, self.stimuli, is_leaf=is_instance(Stimulus))
        # i_clamp = jax.tree.map(i_dist, self.clamps, is_leaf=is_instance(Clamp))

        i_ext_total = jax.tree.reduce(lambda x, y: x + y, i_ext)
        i_int_total = jax.tree.reduce(lambda x, y: x + y, i_int)

        du = jax.tree.map(
            lambda c: c(t, u[c.name], v_at(u, c)),
            self.channels,
            is_leaf=is_instance(Channel),
        )

        dv_ii = (
            i_ext_total - i_int_total
        ) / self.c  # (i_ext / self.area - i_int) / self.c

        # TODO: Fix the units
        # TODO: Add solver for linear system
        i, j = self.edges.T
        dv_ij = self.G(i, j) * (u["v"][j] - u["v"][i])
        dv_ji = self.G(j, i) * (u["v"][i] - u["v"][j])

        du["v"] = dv_ii.at[i].add(dv_ij).at[j].add(dv_ji)  # * 1e3 # mA/cm² -> μA/cm²
        return du

    def set(self, set_dict):
        return tree_set_with_path(self, set_dict)

    def init(self, t, u=None):
        is_channel = lambda x: isinstance(x, Channel)
        v_at = lambda u, c: u["v"][c.index] if u["v"].size > 1 else u["v"]

        u = {} if u is None else u
        u["v"] = jnp.full(self.l.size, -70.0) if "v" not in u else u["v"]
        u0 = jax.tree.map(
            lambda c: c.init(t, u[c.name] if c.name in u else {}, v_at(u, c)),
            self.channels,
            is_leaf=is_channel,
        )
        u0["v"] = u["v"]
        return u0

    def insert(self, mech: Union[Channel, Stimulus, Clamp], at=None):
        if self.l.size > 1:
            at = jnp.arange(self.l.size) if at is None else at
            mech = jax.tree.map(
                lambda *leaves: jnp.stack(leaves)
                if eqx.is_array(leaves[0])
                else leaves[0],
                *[mech] * len(at),
            )
            mech = eqx.tree_at(lambda x: x.index, mech, at)

        if isinstance(mech, Channel):
            channels = self.channels.copy()
            channels.update({mech.name: mech})
            return eqx.tree_at(lambda x: x.channels, self, channels)
        elif isinstance(mech, Stimulus):
            stimuli = self.stimuli.copy()
            stimuli.update({mech.name: mech})
            return eqx.tree_at(lambda x: x.stimuli, self, stimuli)
        elif isinstance(mech, Clamp):
            clamps = self.clamps.copy()
            clamps.update({mech.name: mech})
            return eqx.tree_at(lambda x: x.clamps, self, clamps)
        else:
            raise ValueError(f"Invalid type: {type(mech)}")

    # def remove(self, path: str):
    #     pass

    def train_mask(self, paths: dict[str, Array], init_mask: "Module" = None):
        if init_mask is None:  # all false by default
            set_false = lambda x: jnp.full(x.shape, False) if eqx.is_array(x) else False
            init_mask = jax.tree.map(set_false, self)

        def setter(at):
            def set_true(x):
                return x.at[at].set(True) if at is not None else x.at[:].set(True)

            return set_true

        return tree_apply_with_path(
            init_mask, {p: setter(at) for p, at in paths.items()}
        )

    def share_mask(self, groups: dict[str, Array], init_mask: "Module" = None):
        if init_mask is None:
            default_mask = lambda x: jnp.full(x.shape, 0) if eqx.is_array(x) else 0
            init_mask = jax.tree.map(default_mask, self)

        group_offset = jnp.max(jax.flatten_util.ravel_pytree(init_mask)[0]) + 1

        def group_setter(at, i):
            i += group_offset

            def set_group(x):
                return x.at[at].set(i) if at is not None else x.at[:].set(i)

            return set_group

        return tree_apply_with_path(
            init_mask,
            {p: group_setter(at, i) for i, (p, at) in enumerate(groups.items())},
        )

    # def group(self, index)
    #     pass

    # def at(self, index):
    #     pass

    def pd_render(self):
        df = pd.DataFrame()
        cols = ["l", "r", "c", "Ra"]

        data = []
        for k in cols:
            data.append(np.array(getattr(self, k)))

        for c in self.channels.values():
            for k in c.__annotations__.keys():
                cols.append(c.name + "." + k)
                col = np.full(self.l.size, np.nan)
                col[c.index] = np.array(getattr(c, k))
                data.append(col)

        df = pd.DataFrame(np.column_stack(data), columns=cols)
        return df

    def nx_render(self):
        df = self.pd_render()
        G = nx.Graph()
        G.add_edges_from(np.array(self.edges))
        nx.set_node_attributes(G, df.to_dict(orient="index"))
        return G

    def render(self, backend: str = "pandas"):
        if backend == "pandas":
            return self.pd_render()
        elif backend == "networkx":
            return self.nx_render()
        else:
            raise ValueError(f"Invalid backend: {backend}")
