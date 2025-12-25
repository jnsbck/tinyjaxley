# tinyjaxley

A minimalist implementation of [jaxley](https://github.com/jaxleyverse/jaxley).

The current linecount is:
[linecount.txt]

## todos:
Simulation
- [x] working comps (single compartments)
- [x] working branches (cable)
- [x] working cells (branched morphologies)
    - [ ] add recording API
    - [ ] add state sharing?
    - [ ] add branchpoints (comp with l=0)
    - [ ] rm parents in fav of only edges
- [ ] Add clamping (set) and stimuli (add) for states and currents
- [ ] swc reader
    - [ ] NEURON backend
    - [ ] custom backend
    - [ ] non-cylindrical comps (res_loads, area)
- [ ] working networks (synapses)

Optimization
- [ ] gradients / optimization
    - [ ] add `filter_trainables` (for `eqx.partition`)
    - [ ] add custom optimizer
    - [ ] add selecting trainables API
    - [ ] add parameter sharing
    - [ ] ddd transforms
- [ ] make it fast! (compile and runtime!) (add regression tests)

Infrastructure
- [ ] add tests
    - [ ] match Jaxley
- [ ] add docs

API sugar and BONUS
- [ ] support more flexible indexing, manipulation and viewing
- [ ] jaxley/.nmodl channel transpiler