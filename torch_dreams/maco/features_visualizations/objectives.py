"""
Objective wrapper and utils to build a function to be optimized
"""

import itertools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .losses import dot_cossim
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names

from ..commons import find_layer


def get_layer_name(layer_path: str) -> str:
    return layer_path.split('.')[-1]


def extract_features(model, layer_path, image_size=512):
    
    # Generate a random input tensor
    input_tensor = torch.rand(1, 3, image_size, image_size)

    # Specify the layer whose output you want to extract
    return_nodes = {layer_path: "output"}

    # Create a feature extractor model
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Get the intermediate output
    intermediate_outputs = feature_extractor(input_tensor)

    # Return the output tensor, accessing it by the key assigned in `return_nodes`
    return intermediate_outputs['output']




def get_layer_output_shape(model, layer, image_size=512, for_input=False):
    
    t_dims = None

    def _local_hook(_, _input, _output):
        nonlocal t_dims
        t_dims = _input[0].size() if for_input else _output.size()

    hook = layer.register_forward_hook(_local_hook)
    dummy_var = torch.zeros(1, 3, image_size, image_size)
    model(dummy_var)
    hook.remove()

    return t_dims





class Objective:
    def __init__(self,
                 model: nn.Module,
                 layers: List[nn.Module],
                 masks: List[torch.Tensor],
                 funcs: List[Callable],
                 multipliers: List[float],
                 names: List[str]):
        self.model = model
        self.layers = layers
        self.masks = masks
        self.funcs = funcs
        self.multipliers = multipliers
        self.names = names

    def __add__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an objective.")
        return Objective(
                self.model,
                layers=self.layers + term.layers,
                masks=self.masks + term.masks,
                funcs=self.funcs + term.funcs,
                multipliers=self.multipliers + term.multipliers,
                names=self.names + term.names
        )

    def __sub__(self, term):
        if not isinstance(term, Objective):
            raise ValueError(f"{term} is not an objective.")
        term.multipliers = [-1.0 * m for m in term.multipliers]
        return self + term

    def __mul__(self, factor: float):
        if not isinstance(factor, (int, float)):
            raise ValueError(f"{factor} is not a number.")
        self.multipliers = [m * factor for m in self.multipliers]
        return self

    def __rmul__(self, factor: float):
        return self * factor
  

   
    



    


    @staticmethod
    def layer(model: nn.Module,
              layer: str,
              reducer: str = "magnitude",
              multiplier: float = 1.0,
              name: Optional[str] = None):
        """
        Util to build an objective to maximise a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        reducer
            Type of reduction to apply, 'mean' will optimize the mean value of the
            layer, 'magnitude' will optimize the mean of the absolute values.
        multiplier
            Multiplication factor of the objective.
        name
            A name for the objective.

        Returns
        -------
        objective
            An objective ready to be compiled
        """
        layer_structure = find_layer(model, layer)
        layer_shape = get_layer_output_shape(model, layer_structure)

        get_layer_name(layer)

        mask = np.ones((1, *layer_shape[1:]))

        if name is None:
            name = f"Layer# {layer}"


        power = 2.0 if reducer == "magnitude" else 1.0

        def optim_func(model_output, mask):
            return torch.mean((model_output * mask) ** power)
        

       
        

        return Objective(model, [layer], [mask], [optim_func], [multiplier], [name])
    

    



        


