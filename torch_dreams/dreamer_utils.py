import torch
import numpy as np

def default_func_mean(layer_outputs):
    """Default loss function for torch_dreams

    Args:
        layer_outputs (list): List of layers whose outputs are to be maximized. 

    Returns:
        [torch.tensor]: -loss
    """
    loss = 0.
    for out in layer_outputs:
        loss += out.mean()
    return -loss

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()