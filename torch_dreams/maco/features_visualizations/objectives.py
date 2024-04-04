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
  

   
    
    def compile(self) -> Tuple[nn.Module, Callable, List[str], Tuple]:
        """
        Compile all the sub-objectives into one and return the objects
        for the optimization process.

        Returns
        -------
        model_reconfigured
            Model with the outputs needed for the optimization.
        objective_function
            Function to call that compute the loss for the objectives.
        names
            Names of each objectives.
        input_shape
            Shape of the input, one sample for each optimization.
        """
        # the number of inputs will be the number of combinations possible
        # of the objectives, the mask are used to take into account
        # these combinations
        nb_sub_objectives = len(self.multipliers)

        # re-arrange to match the different objectives with the model outputs
        masks = np.array([np.array(m, dtype=object) for m in itertools.product(*self.masks)])
        masks = [torch.tensor(masks[:, i], dtype=torch.float32) for i in range(nb_sub_objectives)]

        # the name of each combination is the concatenation of each objectives
        names = np.array([' & '.join(names) for names in itertools.product(*self.names)])
        # one multiplier by sub-objective
        multipliers = torch.tensor(self.multipliers)

        def objective_function(model_outputs):
            loss = 0.0
            for output_index in range(0, nb_sub_objectives):
                outputs = model_outputs[output_index]
                loss += self.funcs[output_index](
                    outputs, masks[output_index].to(outputs.device))
                loss *= multipliers[output_index]
            return loss

        # the model outputs will be composed of the layers needed
        model_reconfigured = nn.Sequential(*self.layers)

        nb_combinations = masks[0].shape[0]
        input_shape = (nb_combinations, *model_reconfigured.input_shape[1:])

        return model_reconfigured, objective_function, names, input_shape

    


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
        
        layer_shape = get_layer_output_shape(model, layer)
        layer_output = extract_features(model, layer)
        layer = get_layer_name(layer)
        

        mask = np.ones((1, *layer_shape[1:]))

        if name is None:
            name = f"Layer# {layer}"


        power = 2.0 if reducer == "magnitude" else 1.0

        def optim_func(model_output, mask):
            return torch.mean((model_output * mask) ** power)
        
        return Objective(model, [layer_output], [mask], [optim_func], [multiplier], [name])
    



    @staticmethod
    def channel(model: nn.Module,
                layer: str,
                vectors:Union[torch.tensor,List[torch.tensor]],
                multiplier: float = 1.0,
                cossim_pow: float = 2.0,
                names: Optional[Union[str, List[str]]] = None):
        layer_output = extract_features(model, layer)
        masks = vectors if isinstance(vectors, list) else [vectors]
        
        layer = get_layer_name(layer)

        if names is None:
            names = [f"Direction#{layer}_{i}" for i in range(len(masks))]

        def optim_func(model_output, mask):
            return dot_cossim(model_output, mask, cossim_pow)
        

        return Objective(model, [layer_output], [masks], [optim_func], [multiplier], [names])
    

    @staticmethod
    def direction(model: nn.Module,
                  layer: Union[str],
                  vectors: Union[torch.Tensor, List[torch.Tensor]],
                  multiplier: float = 1.0,
                  cossim_pow: float = 2.0,
                  names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a direction in a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        vectors
            List of vectors to optimize.
        multiplier
            Multiplication factor of the objective.
        cos_sim_pow
            Power of the cosine similarity.    
        names
            A name for the objective.

        Returns
        -------
        objective
            An objective ready to be compiled
        """


        layer_output = extract_features(model, layer)
        masks = vectors if isinstance(vectors, list) else [vectors]

        layer = get_layer_name(layer)

        if names is None:
            names = [f"Direction#{layer}_{i}" for i in range(len(masks))]

        def optim_func(model_output, mask):
            return dot_cossim(model_output, mask, cossim_pow)
        
        return Objective(model, [layer_output], [masks], [optim_func], [multiplier], [names])
    

    @staticmethod
    def channel(model: nn.Module,
                layer: str,
                vectors:Union[torch.tensor,List[torch.tensor]],
                multiplier: float = 1.0,
                cossim_pow: float = 2.0,
                names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a channel in a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        vectors
            List of vectors to optimize.
        multiplier
            Multiplication factor of the objective.
        cos_sim_pow
            Power of the cosine similarity.    
        names
            A name for the objective.

        Returns
        -------
        objective
            An objective ready to be compiled
        """

        layer_output = extract_features(model, layer)
        layer_shape = get_layer_output_shape(model, layer)

        channel_ids = channel_ids if isinstance(channel_ids, list) else [channel_ids]

        masks = np.zeros((len(channel_ids), *layer_shape[1:]))

        for i, channel_id in enumerate(channel_ids):
            masks[i, ..., channel_id] = 1.0

        if names is None:
            names = [f"Channel#{layer}_{i}" for i in channel_ids]


        axis_to_reduce = list(range(1, len(layer_shape)))

        def optim_func(output, target):
            return torch.mean(output * target, axis=axis_to_reduce)
        
        return Objective(model, [layer_output], [masks], [optim_func], [multiplier], [names])
    

    @staticmethod
    def neuron(model:nn.Module,
               layer:str,
               neuron_ids:Union[int,List[int]],
               multiplier:float=1.0,
               names:Optional[Union[str,List[str]]]=None):
        """
        Util to build an objective to maximise a neuron in a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        neuron_ids
            List of neuron ids to optimize.
        multiplier
            Multiplication factor of the objective.
        names
            A name for the objective.

        Returns
        -------
        objective
            An objective ready to be compiled
        """

        layer_output = extract_features(model, layer)
        layer_shape = get_layer_output_shape(model, layer)

        neuron_ids = neuron_ids if isinstance(neuron_ids, list) else [neuron_ids]

        nb_objectives = len(neuron_ids)
        
        
        layer_shape = get_layer_output_shape(model, layer)
        layer_shape = layer_shape[1:]


        masks = np.zeros((nb_objectives, *layer_shape))

        masks = masks.reshape((nb_objectives, -1))
        for i, neuron_id in enumerate(neuron_ids):
            masks[i, neuron_id] = 1.0
        masks = masks.reshape((nb_objectives, *layer_shape))


        if names is None:
            names = [f"Neuron#{layer}_{i}" for i in neuron_ids]

        axis_to_reduce = list(range(1, len(layer_shape)+1))

        def optim_func(output, target):
            return torch.mean(output * target, axis=axis_to_reduce)
        
        return Objective(model, [layer_output], [masks], [optim_func], [multiplier], [names])
    




        













        





    

    



        


