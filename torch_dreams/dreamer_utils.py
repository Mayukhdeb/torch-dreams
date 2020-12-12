import torch
import numpy as np

def default_func_MSE(layer_outputs):
    losses = []
    for output in layer_outputs:

        loss_component = torch.nn.MSELoss(reduction='mean')(output, torch.zeros_like(output))
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    return loss

def default_func_norm(layer_outputs):
    losses = []
    for output in layer_outputs:
        losses.append(output.norm())
    loss = torch.mean(torch.stack(losses))
    return loss

def default_func_mean(layer_outputs):
    losses = []
    for output in layer_outputs:
        losses.append(output.mean())
    loss = torch.mean(torch.stack(losses))
    return loss

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

def get_gradients(net_in, net, layers, default_func, custom_func = None):   
        """
        Executes the forward pass through the model and returns the gradients from the selected layer. 

        input args{
            net_in:  the 3D tensor which is to be used in the forward pass <size = (C, H, W)>
            net:  pytorch model which is being used for the  deep-dream
            layer:  layer instance of net whose activations are to be maximized
        }

        returns{
            gradient of model weights 
        }
        """  
        net_in = net_in.unsqueeze(0)
        net_in.requires_grad = True
        net.zero_grad()

        hooks = []
        for layer in layers:

            hook = Hook(layer)
            hooks.append(hook)

        net_out = net(net_in)

        layer_outputs = []

        for hook in hooks:

            out = hook.output[0]
            layer_outputs.append(out)

        if custom_func is not None:
            loss = custom_func(layer_outputs)
        else:
            loss = default_func(layer_outputs)


        loss.backward()
        return net_in.grad.data.squeeze(0)

def make_octave_sizes(original_size, num_octaves, octave_scale):
    
    sizes = []

    for n in range(-num_octaves, 1):
        octave_size = tuple( np.array(original_size) * octave_scale**n)
        new_size = (int(octave_size[1]), int(octave_size[0]))

        sizes.append(new_size)

    return sizes