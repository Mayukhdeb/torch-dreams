"""
Objective wrapper and utils to build a function to be optimized
"""

import itertools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .losses import dot_cossim
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor

from ..commons import find_layer


def extract_features(model, layer_path, image_size=512):
    """
    Extracts features from a specific layer of a PyTorch model using a randomly generated input tensor.

    Parameters:
    - model (torch.nn.Module): The PyTorch model from which to extract features.
    - layer_path (str): The path to the layer from which to extract features.
    - image_size (int, optional): The size of the image for the automatically generated input tensor. Defaults to 512.

    Returns:
    - The output from the specified layer.
    """
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
    """
    Computes the output shape of the specified layer using a dummy input.

    Parameters:
        model (torch.nn.Module): The PyTorch model.
        layer (torch.nn.Module): The layer to compute the output shape for.
        image_size (int, optional): The size of the image to create the dummy input. Defaults to 512.
        for_input (bool, optional): If True, returns the input shape to the layer, otherwise the output shape.

    Returns:
        torch.Size: The shape of the output (or input) tensor of the layer.
    """
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
    """
    Use to combine several sub-objectives into one.

    Each sub-objective act on a layer, possibly on a neuron or a channel (in
    that case we apply a mask on the layer values), or even multiple neurons (in
    that case we have multiples masks). When two sub-objectives are added, we
    optimize all their combinations.

    e.g Objective 1 target the neurons 1 to 10 of the logits l1,...,l10
        Objective 2 target a direction on the first layer d1
        Objective 3 target each of the 5 channels on another layer c1,...,c5

        The resulting Objective will have 10*5*1 combinations. The first input
        will optimize l1+d1+c1 and the last one l10+d1+c5.

    Parameters
    ----------
    model
        Model used for optimization.
    layers
        A list of the layers output for each sub-objectives.
    masks
        A list of masks that will be applied on the targeted layer for each
        sub-objectives.
    funcs
        A list of loss functions for each sub-objectives.
    multipliers
        A list of multiplication factor for each sub-objectives
    names
        A list of name for each sub-objectives
    """

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
        for the optimisation process.

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
        masks = [torch.stack([m for m in itertools.product(*self.masks)], dim=0)[:, i] for i in
                 range(nb_sub_objectives)]

        # the name of each combination is the concatenation of each objectives
        names = [' & '.join(names) for names in
                 itertools.product(*self.names)]
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
        input_shape = (nb_combinations, *self.model.input_shape)

        return model_reconfigured, objective_function, names, input_shape
    


    @staticmethod
    def layer(model: nn.Module,
              layer: Union[str, int],
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
        layer = find_layer(model, layer)
        layer_shape = get_layer_output_shape(model, layer)

        mask = np.ones((1, *layer_shape[1:]))

        if name is None:
            name = f"Layer# {layer}"


        power = 2.0 if reducer == "magnitude" else 1.0

        def optim_func(model_output, mask):
            return torch.mean((model_output * mask) ** power)
        

         return Objective(model, [layer], [mask], [optim_func], [multiplier], [name])
    

    def direction(model: tf.keras.Model,
                  layer: Union[str],
                  vectors: Union[torch.Tensor, List[torch.Tensor]],
                  multiplier: float = 1.0,
                  cossim_pow: float = 2.0,
                  names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a direction of a layer.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        vectors
            Direction(s) to optimize.
        multiplier
            Multiplication factor of the objective.
        cossim_pow
            Power of the cosine similarity, higher value encourage the objective to care more about
            the angle of the activations.
        names
            A name for each objectives.

        Returns
        -------
        objective
            An objective ready to be compiled
        """
        layer = find_layer(model, layer)
        layer_output =  extract_features(model, layer)
        masks = vectors if isinstance(vectors, list) else [vectors]

        if names is None:
            names = [f"Direction#{layer}_{i}" for i in range(len(masks))]
            

        def optim_func(model_output, mask):
            return dot_cossim(model_output, mask, cossim_pow)
        

        return Objective(model, [layer_output], [masks], [optim_func], [multiplier], [names])



        


