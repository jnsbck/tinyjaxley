from .base import Module
from copy import deepcopy
import pandas as pd
class Network(Module):
    def __init__(self, cells):
        super().__init__(cells)
        self.synapses = []

    @property
    def synapse_states(self):
        synapse_states = []
        for syn in self.synapses:
            synapse_states.append(syn.synapse_states)
        return synapse_states
    
    @property
    def synapse_params(self):
        synapse_params = []
        for syn in self.synapses:
            synapse_params.append(syn.synapse_params)
        return synapse_params

    @property
    def states(self):
        return super().states + self.synapse_states
    
    @property
    def params(self):
        return super().params + self.synapse_params

    def connect(self, comp1, comp2, synapse):
        syn = deepcopy(synapse)
        syn.pre = comp1.index
        syn.post = comp2.index
        self.synapses.append(syn)

    @property
    def edges(self):
        param_states = []
        for params, states in zip(self.synapse_params, self.synapse_states):
            param_states.append({**params, **states})
        return pd.DataFrame(param_states)