# tinyjaxley

A minimalist implementation of [jaxley](https://github.com/jaxleyverse/jaxley).

The current linecount is:
[linecount.txt]

## todos:
Simulation
- [ ] working comps (single compartments)
- [ ] working branches (cable)
- [ ] working cells (branched morphologies)
    - [ ] add recording API
    - [ ] add state sharing?
- [ ] Add clamping (set) and stimuli (add) for states and currents
- [ ] swc reader
    - [ ] NEURON backend
    - [ ] custom backend
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