import torch
from torch import nn
from typing import Tuple, Callable, Union, Optional


def guided_relu_policy(max_value: Optional[float] = None) -> Callable:
    """
    Generate a guided relu activation function.
    Some models have relu with different threshold and plateau settings than a classic relu,
    it is important to preserve these settings when changing activations so that the forward
    remains intact.

    Parameters
    ----------
    max_value
        If specified, the maximum value for the ReLU.
    threshold
        If specified, the threshold for the ReLU.

    Returns
    -------
    guided_relu
        A guided relu activation function.
    """
    relu = nn.ReLU(max_value=max_value, inplace=True)

    def guided_relu(inputs: torch.Tensor) -> Tuple[torch.Tensor, Callable]:
        """
        Guided relu activation function.
        Act like a relu during forward pass, but allows only positive gradients with positive
        activation to pass through during backprop.

        Parameters
        ----------
        inputs
            Input tensor

        Returns
        -------
        output
            Tensor, output or relu transformation.
        grad_func
            Gradient function for guided relu.
        """

        def grad_func(grads):
            gate_activation = inputs > 0.0
            return torch.nn.functional.relu(grads) * gate_activation

        return relu(inputs), grad_func

    return guided_relu




def deconv_relu_policy(max_value: Optional[float] = None) -> Callable:
    """
    Generate a deconv relu activation function.
    Some models have relu with different threshold and plateau settings than a classic relu,
    it is important to preserve these settings when changing activations so that the forward
    remains intact.

    Parameters
    ----------
    max_value
        If specified, the maximum value for the ReLU.
    threshold
        If specified, the threshold for the ReLU.

    Returns
    -------
    deconv_relu
        A deconv relu activation function.
    """
    relu = nn.ReLU(max_value=max_value, inplace=True)

    def deconv_relu(inputs: torch.Tensor) -> Tuple[torch.Tensor, Callable]:
        """
        DeconvNet activation function.
        Act like a relu during forward pass, but allows only positive gradients to pass through
        during backprop.

        Parameters
        ----------
        inputs
            Input tensor

        Returns
        -------
        output
            Tensor, output or relu transformation.
        grad_func
            Gradient function for DeconvNet relu.
        """

        def grad_func(grads):
            return torch.nn.functional.relu(grads)

        return relu(inputs), grad_func

    return deconv_relu




def open_relu_policy(max_value: Optional[float] = None) -> Callable:
    """
    Generate a relu activation function which allows gradients to pass.

    Parameters
    ----------
    max_value
        If specified, the maximum value for the ReLU.
    threshold
        If specified, the threshold for the ReLU.

    Returns
    -------
    open_relu
        A relu which allows all gradients to pass.
    """
    relu = nn.ReLU(max_value=max_value, inplace=True)

    def open_relu(inputs: torch.Tensor) -> Tuple[torch.Tensor, Callable]:
        """
        OpenRelu activation function.
        Act like a relu during forward pass, but allows all gradients to pass through
        during backprop.

        Parameters
        ----------
        inputs
            Input tensor

        Returns
        -------
        output
            Tensor, output or relu transformation.
        grad_func
            Gradient function for OpenRelu.
        """

        def grad_func(grads):
            return grads

        return relu(inputs), grad_func

    return open_relu



def is_relu(layer: nn.Module) -> bool:
    """
    Check if a layer is a ReLU layer

    Parameters
    ----------
    layer
        Layer to check.

    Returns
    -------
    is_relu : bool
        True if the layer is a relu activation.
    """
    return isinstance(layer, nn.ReLU)





def has_relu_activation(layer: nn.Module) -> bool:
    """
    Check if a layer has a ReLU activation.

    Parameters
    ----------
    layer
        Layer to check.

    Returns
    -------
    has_relu
        True if the layer has a relu activation.
    """
    if not hasattr(layer, 'activation'):
        return False
    return layer.activation in [torch.nn.functional.relu, torch.nn.ReLU]




def override_relu_gradient(model: nn.Module, relu_policy: Callable) -> nn.Module:
    """
    Given a model, commute all original ReLU by a new given ReLU policy.

    Parameters
    ----------
    model
        Model to commute.
    relu_policy
        Function wrapped with custom_gradient, defining the ReLU backprop behaviour.

    Returns
    -------
    model_commuted
    """
    cloned_model = model
    cloned_model.load_state_dict(model.state_dict())

    for layer_id in range(len(cloned_model.layers)): # pylint: disable=C0200
        layer = cloned_model.layers[layer_id]
        if has_relu_activation(layer):
            layer.activation = relu_policy()
        elif is_relu(layer):
            max_value = layer.max_value if hasattr(layer, 'max_value') else None
            threshold = layer.threshold if hasattr(layer, 'threshold') else None
            cloned_model.layers[layer_id].call = relu_policy(max_value, threshold)

    return cloned_model




def find_layer(model: nn.Module, layer: Union[str, int]) -> nn.Module:
    """
    Find a layer in a model either by his name or by his index.

    Parameters
    ----------
    model
        Model on which to search.
    layer
        Layer name or layer index

    Returns
    -------
    layer
        Layer found
    """
    if isinstance(layer, str):
        return getattr(model, layer)
    if isinstance(layer, int):
        return model[layer]
    raise ValueError(f"Could not find any layer {layer}.")









