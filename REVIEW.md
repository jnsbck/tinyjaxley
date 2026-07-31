# REVIEW — state/parameter handling & the core abstraction

> Design notes for the `rewrite_state_handling` branch. Collects high-level
> thinking, questions the current choices, and proposes an API. **Nothing here
> is implemented yet.** Numbers in "Evidence" are from benchmarks run on this
> machine (CPU, x64).

## How to read this

- §1–2 restate the goal and the decisions we locked in.
- §3 is the one idea everything hangs on ("everything is a current on a stencil").
- §4–8 are the concrete design: fields/layout → mechanism protocol → lowering →
  solvers → domains → sharing → trainability.
- §9 is a blunt critique of the current code, choice by choice.
- §10–12 are evidence, open questions, and a suggested build order.

---

## 1. Goal & hard requirements

Build a biophysical neuron simulator in JAX that is **fast, flexible, and tiny**
(tinygrad-style: few LOC, powerful abstractions, readable). Adding a new
mechanism / channel / synapse should be obvious and short.

Hard requirements:

1. **Arbitrary shapes.** A state/param can be a scalar-per-comp, a vector, or an
   N-D array per comp.
2. **Sharing across mechanisms.** e.g. a Ca channel writes `cai`, a KCa channel
   reads it; ion reversal potentials shared across channels.
3. **Sharing across compartments.** e.g. all dendritic comps share one `gbar`.
4. **Memory efficiency under sparsity.** A state present on 3 of 1000 comps must
   cost ~3 slots, not 1000. Shared values cost one slot.
5. **Mostly plain JAX.** No equinox. Diffrax solvers are allowed but must not be
   *required* — the core should work with any vector-field solver.

## 2. Decisions locked (from clarifying Q&A)

| Topic | Decision |
|---|---|
| Solver | Want an **implicit** (jaxley-style) solver *and* generic solver-dev support. Expose the pieces a solver needs (`i`, `vf`, `init`, optional `step`); do **not** hardcode a stepper. Any vector-field solver should plug in. |
| Implicit linearization | **Derive it from `i` numerically** (jaxley evaluates `compute_current` at `v` and `v+1e-3`; `modules/base.py:3219`). Mechanisms do **not** hand-write Jacobians. |
| Perf vs flex | **Batch same-type mechanisms under the hood.** Flexible per-mechanism API, but lowering fuses all instances of a type into one vmapped kernel + one fused scatter. |
| Scope | All of: single-/multi-comp cable, **branched morphologies**, **synapses/edges**, **shared cross-mech states**. Layout must be domain-aware from day one. |
| Multi-rate stepping | **Out of scope now.** Keep a single global stepper. Don't design for per-region dt yet. |
| Trainability | **Design-in, don't build.** Keep `p` cleanly partitionable via a trainable mask so `grad`/optax work later. No optimizer this pass. |
| Match jaxley? | **Ballpark only.** Physically correct + self-consistent numerics; no need to replicate jaxley's exact splitting/staggering/unit conversions. The solver can be whatever's clean. |
| State representation | **Open, leaning pytree-of-arrays (SoA)** with `ravel_pytree` at the solver boundary. Decide after benchmarking (§4, §10, §11). |

## 3. The central abstraction: *everything is a current on a stencil*

This is the unifying idea that makes one small core cover channels, axial
coupling, synapses, **and** both explicit and implicit solvers.

A mechanism contributes:

- **State dynamics** `vf`: `dstate/dt` for the states it *owns* (gating vars,
  concentrations). Arbitrary/nonlinear.
- **Currents** `i`: a contribution `i(t, u, p)` that flows into the membrane
  equation of one or more compartments.

Every current has a **stencil** — the (target_comp, source_comps) it couples:

- A channel current couples comp `k` to itself → stencil `(k; k)`.
- An axial current couples comp `k` to its neighbour `j` → stencil `(k; k, j)`.
- A synapse couples post comp `q` to pre comp `p` → stencil `(q; q, p)`.

Given `i` and its stencil, a solver can do **both**:

- **Explicit:** `dv/dt = (Σ i) / cm`; scatter each current into its target and
  step with any vector-field integrator.
- **Implicit:** the linear solve is **over the solve-target variable only**
  (voltage). Linearize each current *in the voltages it reads*, scatter the
  slopes into a sparse matrix `A` at the stencil positions and the offsets into
  `b`, then solve `A v_{n+1} = b`. The stencil *is* the sparsity pattern.

**The implicit step is operator-split, not one big solve.** This is the part the
"everything is a current" framing hides: gating states / concentrations are
**not** in the linear system. A step is staggered (jaxley/NEURON style):

1. advance owned non-voltage states (gates via their own `step`, e.g.
   `m ← m∞ + (m−m∞)·exp(−dt/τ)`; concentrations via `vf`/`step`);
2. assemble `A v_{n+1} = b` from currents linearized in `v` (using the just-
   updated gates);
3. solve for `v`, then loop.

⇒ `step` (§5) is **not optional polish** — it is how the non-voltage half of
every implicit step happens. The *model* exposes `i`, `vf`/`step`, and the
coupling graph; the *solver* owns the splitting scheme (order, staggering).
Getting the split order wrong is the classic "looks right, drifts slowly" bug.

**Sparsity pattern ≠ fast solve.** The stencil gives the pattern of `A`, but not
jaxley's O(N) Hines elimination. Two solver tiers, and they sit exactly on the
speed-vs-generality tension from the opening question:

- *Generic:* scatter conductances into a sparse `A`, hand to a black-box sparse
  solve. Works for **any** topology (ideal for solver-dev), but not O(N).
- *Hines:* needs a topology-aware **comp ordering** + specialized branched-
  tridiagonal elimination. That ordering is **not** carried by per-current
  stencils — it needs the whole **graph**.

⇒ the lowered model must expose the **topology graph explicitly** (reinforces
§8: store edges, not a `parents` array); a solver chooses naive-sparse vs
building a Hines ordering from that graph. "Tree → tridiagonal" is not free.

**Consequence for the API:** channels, axial, and synapses are the *same kind of
object* (a mechanism with `i` + a stencil). We do not need a separate
`EdgeMechanism` hierarchy with its own `vf(u_pre, u_post, ...)` signature — an
edge is just a mechanism whose stencil reads two comps. This is a big
simplification over both the current code and `IMPLEMENTATION_PLAN.md`.

**Linearization for multi-voltage currents.** jaxley's `v`/`v+1e-3` trick
(`modules/base.py:3219`) is for **channel** currents, which read one voltage; it
assembles axial/coupling conductances **analytically** (linear in `v`, geometric
constants) rather than perturbing them. A current reading *two* voltages (axial;
a nonlinear synapse depending on post-`v`) needs `∂i/∂v_pre` **and**
`∂i/∂v_post` — perturb each read voltage separately, or take the conductance
directly when the current is linear in that voltage. So "derive from `i`" must
not be implemented as a single-ε perturbation; the linearizer must know a
current's read-voltages and handle each. Fine as default; a mechanism can
override via `step`.

## 4. Fields & memory layout

Keep the current core idea — **named fields with precomputed integer index maps**
— because it is exactly what delivers arbitrary shapes + sparse storage +
sharing. Clean it up as follows.

> **Open fork — flat vector vs pytree-of-arrays (SoA).** I earlier leaned "one
> flat `u`/`p` per kind"; on reflection that's not decided. Two options:
> - *Flat vector + integer index maps.* Wins: single fused scatter (§10), one
>   `grad`-able array. Costs: raveling gymnastics for N-D states; an implicit ABI.
> - *Pytree/dict of per-state arrays (SoA), what jaxley uses.* Wins: arbitrary
>   shapes and sparse presence are trivial (each entry right-sized); diffrax /
>   `lax.scan` carry pytrees natively; `grad`/partition work with a mask-pytree.
>   Costs: scatter is per-array (may lose the fused-scatter win — **benchmark it**).
>
> Honest framing: **internal representation is likely a pytree of arrays;
> flattening (`ravel_pytree`) is a solver-adapter concern** for solvers that want
> a flat state. Decide after benchmarking SoA-scatter vs flat-fused-scatter
> (§10, §11). The field/index-map machinery below is representation-agnostic.

A **Field** declares one named quantity:

```
Field(ref, value, *, index, group, train)
  ref    "na.m"           full dotted name; last segment is the local name "m"
  value  array            unique rows; row axis is the "storage" axis
  index  int[]            which comps (or edges) this field lives on
  group  int[]            per-entry-in-index → which row of `value` (sharing)
  train  bool[]           trainable mask, aligned to storage rows
```

- **Sparse storage:** `value` has one row per *distinct group*, not per comp. A
  state on 3 comps → 3 rows (or 1 if all share a group). Requirement (4). ✔
- **Sharing across comps** (req. 3): multiple `index` entries map to the same
  `group` row → one stored value, many comps read it. e.g. dendritic `gbar`:
  `group = [0,0,0,...]`.
- **Sharing across mechanisms** (req. 2): identity is `ref`. Two mechanisms
  referencing `cai` resolve to the **same storage slots**; their gather maps
  point at the same offsets. See §8.
- **Arbitrary shape** (req. 1): storage row can be N-D; the flat layout stores
  `value.reshape(n_rows, -1)` contiguously and the index map is 2-D
  (`[n_comps, row_size]`). `dev.ipynb` cell 3 (`x_map`, `z_map`) is the intended
  behaviour and should become the layout's test case.

**Owned vs read-only fields.** Keep the current convention, make it explicit:

- A **`Field` instance** in a mechanism's declaration = *owned* (this mechanism
  allocates & integrates it).
- A **bare string** (e.g. `"v"`, `"cai"`) = *read a field owned elsewhere*
  (global `v`, or another mechanism's state). No allocation.

This one distinction expresses all sharing cleanly and is already half-present in
the code (`Leak.s0 = ("v", ...)`).

**Layout mechanics — question ravel_pytree.** The current code lays out storage
via `ravel_pytree` over a dict and recovers indices by unraveling `arange`.
It works, but: (a) ordering is dict-insertion-order (fragile as an ABI), and (b)
2-D/shaped fields need care. Consider an explicit `Layout` allocator (like the
one sketched in `IMPLEMENTATION_PLAN.md` §2) that assigns each field a
`(start, n_rows, row_shape)` region. It is a few more LOC but makes shaped
states, record plans, and the trainable mask trivial to compute and to reason
about. **Recommendation:** explicit Layout; keep `ravel_pytree` only as the
"combine defaults into the initial `u0/p0`" convenience.

## 5. Mechanism protocol

A mechanism is a small class declaring its fields and up to four methods. The
gathered `u`/`p` passed in are **named tuples** (access `u.v`, `u.m`, `p.gbar`),
built at lowering from the declared local names — readable and zero-cost.

```python
class Na(Channel):
    states = ("v", State("na.m", 0.05), State("na.h", 0.60))  # "v" read-only
    params = (Param("na.gbar", 120.0), Param("na.e", 50.0))
    current = "ina"                     # name of the current it contributes

    def init(self, t, u, p):            # steady state (optional)
        am, bm = a_m(u.v), b_m(u.v); ah, bh = a_h(u.v), b_h(u.v)
        return am/(am+bm), ah/(ah+bh)   # order matches owned states

    def vf(self, t, u, p):              # d(owned states)/dt
        dm = a_m(u.v)*(1-u.m) - b_m(u.v)*u.m
        dh = a_h(u.v)*(1-u.h) - b_h(u.v)*u.h
        return dm, dh

    def i(self, t, u, p):               # membrane current (into `current`)
        return -p.gbar * u.m**3 * u.h * (u.v - p.e)
```

- `i` returns **one array** (the current), not a 1-tuple. The current code wraps
  everything in tuples (`return (-p.gbar*...,)`); drop that — it adds noise. A
  mechanism contributes exactly one named current.
- `vf` returns a tuple **aligned to the owned-state declaration order**. This is
  what lets each owned state keep its own shape (req. 1) — good, keep it.
- Optional **`step(t0, t1, u, p)`**: a mechanism may define how *its own* states
  advance (e.g. exponential Euler for gates: `m ← m∞ + (m−m∞)e^{−dt/τ}`). The
  solver calls `step` when present, else falls back to `vf` + the global
  integrator. This is the seam that (a) matches jaxley's gate handling and
  (b) leaves room for per-mechanism integration later without a multi-rate
  framework now.
- **Stencil** for edges: an edge mechanism declares which comps it reads
  (`pre`/`post` supplied at `connect`), and `u`/`p` gather from both. Same four
  methods; `i`'s target is the post comp. No separate base class needed (§3).

**`is_density` / point vs distributed currents.** Keep the flag (density currents
divide by membrane area; point currents like a clamp do not). But move the
area-divide **out of the model's `vf`** (where it's hardcoded today) into the
current-assembly step so it is uniform for node and edge currents.

## 6. Lowering ("build") — the performance layer

`Model` is a **mutable builder**; `.lower()` produces an **immutable, jittable**
`LoweredModel` (a pytree). Split these two — the current code merges them and
mutates `self` inside `vf`/`build`, which fights JIT and makes the object hard to
pass around. (`IMPLEMENTATION_PLAN.md` had the split; the current `base.py` lost
it.)

Lowering does the expensive, static work **once**:

1. **Collect & resolve fields.** Walk model + all inserted mechanisms; unify by
   `ref`; resolve groups; allocate the Layout; build `u0`, `p0`.
2. **Batch by type.** Group mechanism *instances* by class. All comps carrying
   type `T` become **one kernel**: gather `[n_T, ...]`, `vmap(T.vf)`, `vmap(T.i)`.
   This is the chosen perf strategy — one vmapped call per *type*, not per
   insertion.
3. **Fuse scatters.** Precompute, per kernel, a **single concatenated scatter
   index** for all its owned states, so writing back is one `.at[idx].add(vals)`
   rather than a Python loop of per-state scatters. Evidence (§10) shows this is
   ~3× faster once you have ≳50 comps; only tiny (1-comp) models prefer the loop,
   so default to fused and it's fine.
4. **Build the current-assembly plan.** Per current: its target comps and the
   comps it reads (the stencil). This plan feeds *both* the explicit RHS
   (scatter-add into `dv`) and the implicit assembler (scatter slopes into `A`).
5. **Record plans & trainable mask.** Gather indices for recorded refs; a boolean
   mask over `p` (from `train`) for later `grad`/partition.

Output `LoweredModel` exposes exactly what a solver needs (§7).

## 7. The solver interface (generic, solver-dev friendly)

Solvers must be pluggable and must not live inside the model. The lowered model
exposes a minimal protocol; a solver is any function that consumes it.

```python
class LoweredModel:                     # immutable pytree
    u0: Array; p: Array                 # initial state, params
    def init(self, u, p) -> u           # steady-state (calls mechs' init)
    def rhs(self, t, u, p) -> du         # explicit vector field (Σ currents + vf)
    def currents(self, t, u, p) -> ...   # per-current values + stencils
    topology: Graph                      # comp-comp edges (for Hines ordering)
    def record(self, u) -> dict          # gather recorded refs
```

- **Explicit solvers** (forward Euler, RK, diffrax, custom) use only `rhs`. Since
  `rhs` is a plain `f(t, u, p)`, diffrax `ODETerm`, adaptive explicit steppers,
  and dense/interpolated output all work with zero coupling to diffrax in the
  core. Requirement (5). ✔
- **Implicit / Hines solver** is **operator-split** (§3): first advance
  non-voltage states via each mechanism's `step`/`vf`, then linearize the
  currents in `v` (`i(v)`, `i(v+ε)`; per-read-voltage for multi-voltage
  currents), scatter slopes into a sparse `A` + offsets into `b`, solve for `v`.
  The stencil supplies the sparsity *pattern*; the O(N) Hines solve additionally
  needs the topology **graph** (exposed by the lowered model) to build the
  elimination order. A generic solver can skip that and use a black-box sparse
  solve for any topology.
- A solver is `solve(lowered, stepper, ts, p) -> trajectory` with a `lax.scan`.
  Custom solver development = write a new `stepper(lowered, u, p, t0, t1) -> u`.

**Adaptive / interpolation:** falls out of the explicit path for free (diffrax or
a custom controller on `rhs`). Implicit-adaptive and dense output are future work;
the interface doesn't preclude them. **Multi-rate is explicitly deferred** — but
note the `step`-per-mechanism seam + state-slicing layout are the hooks that would
enable it later without an API break.

## 8. Sharing, domains, branched topology, edges

- **Shared cross-mechanism state (req. 2), the calcium example.** `cai` is owned
  once (by a `CaConc` mechanism, or declared on the model). `CaHVA` *reads* `v`,
  *reads* `cai`, and contributes to `cai`'s dynamics or a Ca current; `KCa`
  *reads* `cai` and `v`. Because identity is `ref`, all three resolve to the same
  `cai` slots. Owner's `vf` integrates `cai`; others just read. This is the
  cleanest form and needs the "bare string = read shared" convention (§4) plus a
  rule for *who owns* a shared, writable state (exactly one owner; others
  read-only). **Open design point:** what if two mechanisms both want to *add* to
  `cai` (e.g. two Ca currents feeding one pool)? Then `cai`'s dynamics must sum
  contributions — model it as a current-like accumulation into the pool, i.e. the
  same "current on a stencil" pattern (§3) targeting a concentration instead of
  voltage. Worth prototyping.
- **Domains.** Fields live on a domain: `comp` (node) or `edge`. The layout is
  domain-tagged; `v`, `cai`, gates are `comp`; synaptic weights are `edge`. Edge
  currents target a `comp` (post). Keep it to these two domains for now (a
  `global` domain for scalars is a trivial extension).
- **Branched morphology.** Topology is a set of comp-to-comp edges (parent/child).
  Axial coupling is just edge mechanisms over those edges (§3), so branched cells
  need *no new machinery* beyond the edge domain + a solver that can assemble the
  tree's sparse system. Recommend storing topology as explicit edges (not a
  `parents` array) — it unifies cable, tree, and network.
- **Synapses/networks** are edge mechanisms across cells — same abstraction as
  axial, different `i`. Population connectivity = many edges; batch by type (§6).

## 9. Critique of the current implementation (questioning each choice)

Read against `tinyjaxley/` on this branch. It is mid-surgery and does not run
(`build()` → `KeyError: 'hh.m'`), so this is about direction, not bugs.

1. **`insert()` doesn't register fields (broken core).** The merge of mechanism
   states/params into the global registry is commented out, so `build()` can't
   find `hh.m`. *Fix:* register at insert **or** (cleaner) collect everything
   lazily in `.lower()`. **Recommend lazy collection in lower** — insert stays a
   pure spec, lowering is the single place that sees the whole model.
2. **`Model` mutates itself and rebuilds inside `vf`.** `vf` calls `self.build()`
   on the fly and references `self.i_to_v`/`i_to_cap` that `build()` no longer
   sets. Split builder vs `LoweredModel` (§6); make the lowered object a pure
   pytree with no hidden rebuilds.
3. **Mechanisms keyed by `name` in a dict → repeated inserts overwrite.**
   `self.mechanisms[mech.name] = mech`. Inserting HH on `[0,1]` then `[2,3]` loses
   the first. With type-batching (§6) the natural key is *type*, and inserts of
   the same type should **merge comp indices**, not replace. `merge_fields`
   ("last wins per comp") exists but isn't wired here.
4. **Everything wrapped in 1-tuples** (`return (-p.gbar*...,)`, `i0 = (Current,)`).
   Currents are singular — return the array. Reserve tuples for `vf`'s
   multiple owned states (where they earn their keep by allowing per-state shapes).
5. **Currents stored as `Field`s with defaults (`i0`).** Currents are *derived*
   each step, not integrated state. Storing default values for them is confusing.
   Keep a currents *buffer* (scatter target) computed per step; only *record* it if
   asked. (Jaxley recomputes; don't persist.)
6. **Area/cap divide hardcoded in `Model.vf`.** `du.at[i_to_v].add(currents/cap)`
   with special `i_to_*` maps. Move into current-assembly so node & edge currents
   are handled uniformly and `v`/`cap`/`rad`/`len` aren't privileged by string.
   (`v` *is* legitimately special as the solve target — document that, but don't
   scatter it ad hoc.)
7. **`ravel_pytree`-based layout is an implicit ABI.** Works, but see §4 — prefer
   an explicit `Layout` for shaped states, record plans, and the trainable mask.
8. **`EdgeMechanism` with a separate `vf(u_pre, u_post, ...)` signature**
   (in `IMPLEMENTATION_PLAN.md`) — drop it. Unify under "mechanism + stencil" (§3);
   far fewer LOC and one code path for the solver.
9. **`gather` builds a namedtuple per kernel call.** Fine *after* type-batching
   (one gather per type per step). Verify it doesn't allocate in the hot loop; if
   it does, precompute the namedtuple class at lowering and only fill it.

**What to keep (it's the good part):** the flat-vector + precomputed-index-map
model; `Field` with `ref`/`index`/`group`/`train`; owned-vs-string distinction;
`_vtrap`/`safe_exp` numerics; the build/run split *intent*; the benchmark harness.

## 10. Evidence (benchmarks run here)

- **Fused vs looped scatter** (`benchmarks/scatter_add.py`, 10k steps):
  - `ncomp=1`: loop 0.079 ms; concat 0.974 ms; preflat 1.089 ms → **loop wins**
    (concat overhead dominates for tiny state).
  - `ncomp=50`: loop 22.8 ms; concat 8.4 ms; **preflat 7.2 ms → 3.18× faster**.
  - ⇒ Precompute one concatenated scatter index at lowering and scatter once
    (§6.3). Only degenerate 1-comp models prefer the loop; not worth special-casing.
- **Current impl doesn't run** (`uv run pytest` → `KeyError: 'hh.m'`), confirming
  the registry gap in §9.1. The regression test's hand-written `_baseline_vf`
  encodes the target semantics and the "ceiling" we should match/approach.
- Not yet measured but should be, before committing: (a) type-batched vmap kernel
  vs the hand-written monolithic `vf` (the `test_regression` baseline) — how much
  does gather+vmap+scatter cost vs a fused expression at ncomp = 1/50/1000?
  (b) **SoA-scatter (into each state's own array) vs flat-fused-scatter** — the
  fork in §4; the benchmark above only covered flat-vector scatter, not SoA.
  (c) implicit Hines step time vs jaxley on a branched cell (ballpark target).

## 11. Open questions / risks

1. **Who owns a shared writable state, and how do multiple writers accumulate?**
   (§8, the two-Ca-currents-into-one-pool case.) Prefer modelling accumulation as
   "current on a stencil" targeting the pool; needs a prototype.
2. **Type-batching with heterogeneous params.** Two `Na` inserts with different
   `gbar` batch fine (params are per-comp rows). But two inserts with different
   *group* structure (one shared, one per-comp) must still batch — confirm the
   index-map construction handles mixed groups within a type.
3. **Numerical linearization cost.** Evaluating every current twice per implicit
   step doubles `i` calls. Jaxley accepts this. Measure; consider caching or
   analytic conductance for the hot channels only.
4. **`step`-override composition.** If some mechanisms define `step` and others
   don't, the global solver must interleave "solve implicit voltage" with
   "advance gates by their own step" correctly (operator splitting, as in
   jaxley/NEURON). Define the split explicitly.
5. **Trainable mask through shaped/shared fields.** A shared param is one storage
   slot; marking it trainable must produce a mask over storage, and `grad` must
   flow back through the gather to that single slot. Verify with an autodiff test.
6. **Diffrax + `lax.scan` recording.** Decide the recording API (SaveAt-like)
   independent of solver so explicit/implicit/diffrax all record uniformly.
   Recording a *current* means recomputing it (currents aren't state) → the
   step/scan must surface a currents buffer as aux output.
7. **State representation (flat vs SoA).** §4 fork. Blocks committing the layout;
   resolve with the §10(b) benchmark early.
8. **Operator-split order.** With per-mechanism `step` overrides mixed with the
   voltage solve, define the split/stagger explicitly (§3). Ballpark accuracy is
   the bar, but the split must still be *consistent* or slow drift creeps in.

## 12. Suggested build order (keep each step runnable)

1. **Layout + Field** (explicit `Layout`, shaped + shared + sparse) — unit-test
   against `dev.ipynb` cell 3 (`v/n/m/h/x/z` maps).
2. **Mechanism protocol** (`states/params/current`, `i/vf/init/step`,
   owned-vs-string) — HH, Na, K, Leak, StepCurrent, no tuples-on-currents.
3. **Lower**: collect → Layout → type-batched kernels → fused scatter →
   current-assembly plan. Produce immutable `LoweredModel`.
4. **Explicit solver** on `rhs` (forward Euler + a diffrax adapter) → reproduce
   `test_regression` single-comp HH; then compare to jaxley in `test_match_jaxley`
   at **ballpark** tolerance (same spike times/shape, not bit-for-bit).
5. **Edges/stencils + axial** as a mechanism → multi-comp cable.
6. **Implicit Hines solver** from `currents` (numerical linearization) → branched
   cell; benchmark vs jaxley.
7. **Shared cross-mech state** (Ca pool + KCa) → validates §8.
8. Trainable mask + a `grad` smoke test (no optimizer).

Milestones 1–4 are the minimum to prove the abstraction; 5–6 prove it scales to
real morphologies; 7–8 prove flexibility and the optimization seam.
