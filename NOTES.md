# Goals
- Speed, simplicity, flexibility and minimal LOCs

# Design decisions
- Build and run phase
    - needed to be able to share states, params and currents between mechanisms
- Fields
    - params, states, currents, auxillary variables that are local are "fields"
    - fields can have any shape
    - fields track their position and are considered the same if f1.ref == f2.ref -> sharing between mechanisms
    - fields also track how they are shared via "groups" -> sharing between compartments
- Domains
    - global, node, edge
    - mechansims can exist on each domain
- Mechanisms
    - vf, init, i, step methods that accept and return tuples of states / params
    - returning tuples allows states to have different shapes, although at the cost of performance
