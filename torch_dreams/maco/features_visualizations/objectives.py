import torch
import torch.nn as nn
import itertools
import numpy as np
from ..commons import get_module_by_name
from .losses import dot_cossim

def get_tensor_dimensions_impl(model, layer_path, image_size, for_input=False):
    t_dims = None
    layer = model
    for attr in layer_path.split('.'):
        if attr.isdigit():
            layer = layer[int(attr)]
        else:
            layer = getattr(layer, attr)

    def _local_hook(_, _input, _output):
        nonlocal t_dims
        t_dims = _input[0].size() if for_input else _output.size()

    hook = layer.register_forward_hook(_local_hook)
    dummy_var = torch.zeros(1, 3, image_size, image_size)
    model(dummy_var)
    hook.remove()

    return t_dims


def extract_features(input_tensor, model, layer_path):
    # Specify the layer whose output you want to extract
    return_nodes = {
        layer_path: "output"
    }

    # Create a feature extractor model
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Get the intermediate output
    intermediate_outputs = feature_extractor(input_tensor)

    # Return the output tensor directly, accessing it by the key assigned in `return_nodes`
    return intermediate_outputs['output']




class Objective:
    def __init__(self, model, layers, masks, funcs, multipliers, names):
        self.model = model
        self.layers = layers
        self.masks = masks
        self.funcs = funcs
        self.multipliers = multipliers
        self.names = names

    def __add__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an Objective.")
        return Objective(
            self.model,
            self.layers + term.layers,
            self.masks + term.masks,
            self.funcs + term.funcs,
            self.multipliers + term.multipliers,
            self.names + term.names
        )

    def __sub__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an Objective.")
        term.multipliers = [-1.0 * m for m in term.multipliers]
        return self + term

    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise ValueError(f"{factor} is not a number.")
        self.multipliers = [m * factor for m in self.multipliers]
        return self

    def __rmul__(self, factor):
        return self * factor

    

    

    


    