import torch
from torch import nn
from typing import Tuple, Callable, Union, Optional
from functools import reduce

def open_relu_policy(max_value=None, threshold=0.0):
    """
    Corrected version of the open ReLU policy for PyTorch.

    Parameters
    ----------
    max_value : float, optional
        The maximum value to clamp the output to. If None, no clamping is applied. Default is None.
    threshold : float, optional
        The threshold value for the ReLU function. Values below this threshold are set to 0. Default is 0.0.

    Returns
    -------
    function
        A function that applies the open ReLU operation to its input tensor.
    """
    def open_relu(input_tensor):
        """
        Applies the open ReLU operation to the input tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to apply the open ReLU operation to.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the open ReLU operation.
        """
        # Apply threshold condition
        output = torch.where(input_tensor > threshold, input_tensor, torch.tensor(0.0))
        # Apply max_value condition if specified
        if max_value is not None:
            output = torch.clamp(output, max=max_value)
        return output

    return open_relu



def is_relu(layer: nn.Module) -> bool:
    """
    Check if a layer is a ReLU layer in PyTorch

    Parameters
    ----------
    layer : nn.Module
        Layer to check.

    Returns
    -------
    is_relu : bool
        True if the layer is a relu activation.
    """
    return isinstance(layer, nn.ReLU)





# def has_relu_activation(layer: nn.Module) -> bool:
#     """
#     Check if a layer has a ReLU activation.

#     Parameters
#     ----------
#     layer
#         Layer to check.

#     Returns
#     -------
#     has_relu
#         True if the layer has a relu activation.
#     """
#     if not hasattr(layer, 'activation'):
#         return False
#     return layer.activation in [torch.nn.functional.relu, torch.nn.ReLU]




# def override_relu_gradient(model: nn.Module, relu_policy: Callable) -> nn.Module:
#     """
#     Given a model, commute all original ReLU by a new given ReLU policy.

#     Parameters
#     ----------
#     model
#         Model to commute.
#     relu_policy
#         Function wrapped with custom_gradient, defining the ReLU backprop behaviour.

#     Returns
#     -------
#     model_commuted
#     """
#     cloned_model = model
#     cloned_model.load_state_dict(model.state_dict())

#     for layer_id in range(len(cloned_model.layers)): # pylint: disable=C0200
#         layer = cloned_model.layers[layer_id]
#         if has_relu_activation(layer):
#             layer.activation = relu_policy()
#         elif is_relu(layer):
#             max_value = layer.max_value if hasattr(layer, 'max_value') else None
#             threshold = layer.threshold if hasattr(layer, 'threshold') else None
#             cloned_model.layers[layer_id].call = relu_policy(max_value, threshold)

#     return cloned_model




def get_module_by_name(model: nn.Module, name: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.

    Args:
        module (Union[TensorType, nn.Module]): module whose submodule you want to access
        name (str): the string representation of the submodule. Like ⁠ "module.something.this_thing" ⁠

    Returns:
        object: module that you wanted to extract
    """
    if name != "":
        names = name.split(sep=".")
    else:
        return model
    return reduce(getattr, names, model)









