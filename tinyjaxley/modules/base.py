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

from typing import Optional


class Module(eqx.Module):
    l: Array = eqx.field(converter=jnp.array)
    r: Array = eqx.field(converter=jnp.array)
    c: Array = eqx.field(converter=jnp.array)
    # Ra: Array = eqx.field(converter=jnp.array)
    xyz: Array = eqx.field(converter=jnp.array)
    channels: dict[str, Channel]
    stimuli: dict[str, Stimulus]  # add to state or current
    clamps: dict[str, Clamp]  # set state or current
    # synapses: List[Synapse]

    parent: Array = eqx.field(converter=jnp.array)
    index: Array = eqx.field(converter=jnp.array)
    id: Array = eqx.field(converter=jnp.array)

    edges: Array = eqx.field(converter=jnp.array)

    def __init__(
        self,
        l: Array = 10.0,
        r: Array = 1.0,
        c: Array = 1.0,
        # Ra: Array = 10_000.0,
        key: str = None,
    ):
        self.l = l
        self.r = r
        self.c = c
        # self.Ra = Ra
        self.channels = {}
        self.stimuli = {}
        self.clamps = {}
        self.xyz = jnp.array([0.0, 0.0, 0.0])

        self.parent = jnp.array(-1)
        self.index = jnp.array(0)
        self.id = jnp.array(0)

        # TODO: Add post init to initialize edges
        self.edges = self._edges_init()  # needs to be rerun if parents change

    @property
    def area(self):
        return 2.0 * jnp.pi * self.r * self.l  # um²

    def _edges_init(self):
        if self.parent.size > 1:
            edges = jnp.stack([self.parent, self.index], axis=1)
            edges = edges[edges[:, 0] != -1]
            edges = jnp.concatenate([edges, edges[:, ::-1]])
            return edges
        else:
            return jnp.array([]).reshape(0, 2)

    # TODO: Add branchpoints
    # def _insert_branchpoints(self):
    #     pass

    def G(self, i, j):
        """
        from `https://en.wikipedia.org/wiki/Compartmental_neuron_models`.
        `radius`: um, `Ra`: ohm cm, `l`: um, `g`: mS / cm^2
        """
        r_i, r_j = self.r[i], self.r[j]
        Ra_i, Ra_j = self.Ra[i], self.Ra[j]
        l_i, l_j = self.l[i], self.l[j]
        g = r_i * r_j**2 / (Ra_i * r_j**2 * l_i + Ra_j * r_i**2 * l_j) / l_i
        return g * 1e7  # S/cm/um -> mS / cm²

    def __call__(self, t, u, args=None):
        # TODO: Use seperate diffrax.ODETerms for channels and comp??
        is_instance = lambda cls: lambda x: isinstance(x, cls)
        v_at = lambda u, idx: u["v"][idx] if u["v"].size > 1 else u["v"]

        # TODO: Add clamping
        # TODO: Add state sharing 

        i0 = jnp.zeros(self.l.size)

        def i_dist(c):
            v = v_at(u, c.index)
            u_ = u[c.name] if c.name in u else {}
            i = c.i(t, u_, v)
            return i0.at[c.index].set(i)

        i_int = jax.tree.map(i_dist, self.channels, is_leaf=is_instance(Channel))
        i_ext = jax.tree.map(i_dist, self.stimuli, is_leaf=is_instance(Stimulus))
        # i_clamp = jax.tree.map(i_dist, self.clamps, is_leaf=is_instance(Clamp))

        i_ext_total = jax.tree.reduce(lambda x, y: x + y, i_ext)
        i_int_total = jax.tree.reduce(lambda x, y: x + y, i_int)

        du = jax.tree.map(
            lambda c: c(t, u[c.name], v_at(u, c.index)),
            self.channels,
            is_leaf=is_instance(Channel),
        )

        dv_ii = (i_ext_total  * 1e5 / self.area - i_int_total * 1e3) / self.c

        # TODO: Add solver for linear system
        i, j = self.edges.T
        # bcco = (i, j, self.G(i, j))
        dv_ij = self.G(i, j) * (u["v"][j] - u["v"][i])

        du["v"] = dv_ii.at[i].add(dv_ij)
        return du

    # def step(self, t0, t1, u0):
    #     dt = t1 - t0
    #     v_at = lambda u, idx: u0["v"][idx] if u0["v"].size > 1 else u0["v"]
    #     is_instance = lambda cls: lambda x: isinstance(x, cls)

    #     i0 = jnp.zeros(self.l.size)

    #     def i_dist(c):
    #         v = v_at(u0, c.index)
    #         u_ = u0[c.name] if c.name in u0 else {}
    #         i = c.i(t0, u_, v)
    #         return i0.at[c.index].set(i)

    #     i_int = jax.tree.map(i_dist, self.channels, is_leaf=is_instance(Channel))
    #     i_ext = jax.tree.map(i_dist, self.stimuli, is_leaf=is_instance(Stimulus))
    #     # i_clamp = jax.tree.map(i_dist, self.clamps, is_leaf=is_instance(Clamp))

    #     def step_gate(c):
    #         _u = u0[c.name] if c.name in u0 else {}
    #         _v = v_at(u0, c.index)
    #         xinf = c.xinf(_u, _v)
    #         tau = c.tau(_u, _v)
    #         return exp_euler(_u, dt, xinf, tau)

    #     u1_gates = jax.tree.map(step_gate, self.channels, is_leaf=is_instance(Channel))
    #     u1 = {**u1_gates}

    #     def step_v():

        


    def set(self, set_dict):
        return tree_set_with_path(self, set_dict)

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

    # def group(self, index)
    #     pass

    # def at(self, index):
    #     pass

    # def __iter__(self):
    #     pass

    # def __getitem__(self, index):
    #     pass

    # def render_with_pandas(self):
    #     df = pd.DataFrame()
    #     cols = ["id", "l", "r", "c", "Ra"]

    #     data = []
    #     for k in cols:
    #         data.append(np.array(getattr(self, k)))

    #     for c in self.channels.values():
    #         for k in c.__annotations__.keys():
    #             cols.append(c.name + "." + k)
    #             col = np.full(self.l.size, np.nan)
    #             col[c.index] = np.array(getattr(c, k))
    #             data.append(col)

    #     data = np.column_stack(data)
    #     df = pd.DataFrame(
    #         data, columns=cols, index=np.array(jnp.atleast_1d(self.index))
    #     )
    #     return df

    # def render_as_string(self, show_attrs: bool = True):
    #     G = self.render_with_networkx()
    #     D = nx.DiGraph()
    #     D.add_edges_from(np.array(self.edges))
    #     nx.set_node_attributes(D, G.nodes)

    #     roots = sorted([n for n in D.nodes() if D.in_degree(n) == 0])

    #     type_dict = {0: "undefined", 1: "soma", 2: "axon", 3: "dendrite"}

    #     def _dfs(node, prefix="", is_last=True):
    #         lines, attrs = [], D.nodes[node]
    #         fmt_attr = lambda k, v: (
    #             f"{k}={v:.1f}" if isinstance(v, (int, float)) else f"{k}={v}"
    #         )
    #         attrs_str = (
    #             ", ".join(
    #                 fmt_attr(k, v) for k, v in sorted(attrs.items()) if pd.notna(v)
    #             )
    #             if show_attrs
    #             else ""
    #         )
    #         attrs_part = f"({attrs_str})" if attrs_str else ""
    #         label = f"{type_dict[G.nodes[node]['id']]}[{node}]{attrs_part}"
    #         children = sorted(D.successors(node))
    #         is_root = node in roots
    #         symbol = "" if is_root else ("└── " if is_last else "├── ")
    #         lines.append(f"{prefix}{symbol}{label}{'/' if children else ''}")
    #         for i, child in enumerate(children):
    #             child_is_last = i == len(children) - 1
    #             if is_root:
    #                 next_prefix = ""
    #             else:
    #                 next_prefix = prefix + ("    " if is_last else "│   ")
    #             lines.extend(_dfs(child, next_prefix, child_is_last))
    #         return lines

    #     return "\n".join(line for r in roots for line in _dfs(r, "", r == roots[-1]))

    # def render_with_networkx(self):
    #     df = self.render_with_pandas()
    #     G = nx.Graph()
    #     G.add_edges_from(np.array(self.edges))
    #     nx.set_node_attributes(G, df.to_dict(orient="index"))
    #     return G

    # def render(self, backend: str = "pandas"):
    #     if backend == "pandas":
    #         return self.render_with_pandas()
    #     elif backend == "networkx":
    #         return self.render_with_networkx()
    #     elif backend == "string":
    #         return print(self.render_as_string())
    #     else:
    #         raise ValueError(f"Invalid backend: {backend}")
