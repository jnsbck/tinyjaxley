# tinyjaxley

A minimalist implementation of [jaxley](https://github.com/jaxleyverse/jaxley).

The current linecount is:
[linecount.txt]

## todos:
Simulation
- [x] working comps (single compartments)
- [x] working branches (cable)
- [x] working cells (branched morphologies)
    - [ ] rm parents in fav of only edges
    - [x] set, get, insert API
    - [ ] add state sharing?
    - [ ] add branchpoints (comp with l=0)
- [x] Custom BackwardEuler Solver
    - [x] adaptive timestepping
    - [ ] integrate convenience function
- [ ] Add clamping (set) and stimuli (add) for states and currents
- [ ] add recording API
- [ ] swc reader
    - [ ] NEURON backend
    - [ ] custom backend
    - [ ] non-cylindrical comps (res_loads, area)
- [ ] networks (synapses)

Optimization
- [ ] gradients / optimization
    - [ ] add `filter_trainables` (for `eqx.partition`)
    - [ ] add selecting trainables API
    - [ ] add parameter sharing
    - [ ] add transforms
    - [ ] add custom optimizer
- [ ] make it fast! (compile and runtime!) (add regression tests)

Infrastructure
- [ ] add tests
    - [ ] match Jaxley
- [ ] add docs

API sugar and BONUS
- [x] support more flexible indexing, manipulation and viewing
- [ ] jaxley/.nmodl channel transpiler