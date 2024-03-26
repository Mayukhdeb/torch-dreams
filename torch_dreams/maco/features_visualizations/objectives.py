import itertools
import torch
import numpy as np
from ..commons import get_module_by_name
from ..types import Union, List, Callable, Tuple, Optional
from .losses import dot_cossim



    
import torch
import torch.nn as nn
from itertools import product

class Objective:
    """
    Use to combine several sub-objectives into one.

    Parameters
    ----------
    model : torch.nn.Module
        Model used for optimization.
    layers : List[str]
        A list of the names of layers output for each sub-objectives.
    masks : List[torch.Tensor]
        A list of masks that will be applied on the targeted layer for each sub-objectives.
    funcs : List[Callable]
        A list of loss functions for each sub-objectives.
    multipliers : List[float]
        A list of multiplication factor for each sub-objectives
    names : List[str]
        A list of name for each sub-objectives
    """

    def __init__(self, model, layers, masks, funcs, multipliers, names):
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

    def __mul__(self, factor):
        if not isinstance(factor, (int, float)):
            raise ValueError(f"{factor} is not a number.")
        self.multipliers = [m * factor for m in self.multipliers]
        return self

    def __rmul__(self, factor):
        return self * factor

    def compile(self):
        """
        Compile all the sub-objectives into one and return the objects
        for the optimisation process.

        Returns
        -------
        model_with_hooks : torch.nn.Module
            Model with hooks added to capture the required layers' outputs.
        objective_function : Callable
            Function to call that compute the loss for the objectives.
        names : List[str]
            Names of each objectives.
        input_shape : Tuple
            Shape of the input, one sample for each optimization.
        """

        layer_outputs = {}

        def get_hook(name):
            def hook(model, input, output):
                layer_outputs[name] = output
            return hook

        for name in self.layers:
            layer = dict([*self.model.named_modules()])[name]
            layer.register_forward_hook(get_hook(name))

        def objective_function(input_tensor):
            self.model(input_tensor)
            loss = 0.0
            for i, (func, mask, multiplier) in enumerate(zip(self.funcs, self.masks, self.multipliers)):
                output = layer_outputs[self.layers[i]]
                masked_output = output * mask
                loss += multiplier * func(masked_output)
            return loss

        input_shape = next(self.model.parameters()).shape

        return self.model, objective_function, self.names, input_shape

    @staticmethod
    def layer(model, layer_name, reducer="magnitude", multiplier=1.0, name=None):
        """
        Util to build an objective to maximise a layer.

        Parameters
        ----------
        model : torch.nn.Module
            Model used for optimization.
        layer_name : str
            Name of the targeted layer.
        reducer : str
            Type of reduction to apply.
        multiplier : float
            Multiplication factor of the objective.
        name : Optional[str]
            A name for the objective.

        Returns
        -------
        objective : Objective
            An objective ready to be compiled
        """
        layer = dict([*model.named_modules()])[layer_name]
        layer_shape = tuple(layer.weight.size()) if hasattr(layer, 'weight') else layer.output_shape

        mask = torch.ones(1, *layer_shape[1:])

        if name is None:
            name = f"Layer#{layer_name}"

        def optim_func(output, mask):
            if reducer == "magnitude":
                return torch.mean((output * mask) ** 2)
            else:
                return torch.mean(output * mask)

        return Objective(model, [layer_name], [mask], [optim_func], [multiplier], [name])





   

    @staticmethod
    def direction(model: torch.nn.Module,
                  layer: Union[str, int],
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
        layer = get_module_by_name(model, layer)
        masks = vectors if isinstance(vectors, list) else [vectors]

        if names is None:
            names = [f"Direction#{layer.name}_{i}" for i in range(len(masks))]

        def optim_func(model_output, mask):
            return dot_cossim(model_output, mask, cossim_pow)

        return Objective(model, [layer], [masks], [optim_func], [multiplier], [names])


    

    @staticmethod
    def channel(model: torch.nn.Module,
                layer: Union[str, int],
                channel_ids: Union[int, List[int]],
                multiplier: float = 1.0,
                names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a channel.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        channel_ids
            Indexes of the channels to maximise.
        multiplier
            Multiplication factor of the objectives.
        names
            Names for each objectives.

        Returns
        -------
        objective
            An objective containing a sub-objective for each channels.
        """
        layer = get_module_by_name(model, layer)
        layer_shape = layer.shape
        channel_ids = channel_ids if isinstance(channel_ids, list) else [channel_ids]

        # for each targeted channel, create a boolean mask on the layer to target
        # the channel
        masks = np.zeros((len(channel_ids), *layer_shape[1:]))
        for i, c_id in enumerate(channel_ids):
            masks[i, ..., c_id] = 1.0

        if names is None:
            names = [f"Channel#{layer.name}_{ch_id}" for ch_id in channel_ids]

        axis_to_reduce = list(range(1, len(layer_shape)))

        def optim_func(output, target):
            return torch.mean(output * target, axis=axis_to_reduce)

        return Objective(model, [layer], [masks], [optim_func], [multiplier], [names])



    @staticmethod
    def neuron(model: torch.nn.Module,
               layer: Union[str, int],
               neurons_ids: Union[int, List[int]],
               multiplier: float = 1.0,
               names: Optional[Union[str, List[str]]] = None):
        """
        Util to build an objective to maximise a neuron.

        Parameters
        ----------
        model
            Model used for optimization.
        layer
            Index or name of the targeted layer.
        neurons_ids
            Indexes of the neurons to maximise.
        multiplier
            Multiplication factor of the objectives.
        names
            Names for each objectives.

        Returns
        -------
        objective
            An objective containing a sub-objective for each neurons.
        """
        layer = get_module_by_name(model, layer)

        neurons_ids = neurons_ids if isinstance(neurons_ids, list) else [neurons_ids]
        nb_objectives = len(neurons_ids)
        layer_shape = layer.shape[1:]

        # for each targeted neurons, create a boolean mask on the layer to target it
        masks = np.zeros((nb_objectives, *layer_shape))
        masks = masks.reshape((nb_objectives, -1))
        for i, neuron_id in enumerate(neurons_ids):
            masks[i, neuron_id] = 1.0
        masks = masks.reshape((nb_objectives, *layer_shape))

        if names is None:
            names = [f"Neuron#{layer.name}_{neuron_id}" for neuron_id in neurons_ids]

        axis_to_reduce = list(range(1, len(layer_shape)+1))

        def optim_func(output, target):
            return torch.mean(output * target, axis=axis_to_reduce)

        return Objective(model, [layer], [masks], [optim_func], [multiplier], [names])