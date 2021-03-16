import torch 
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from ..dreamer_utils import Hook, default_func_mean

from .utils import (
    init_image_param, 
    fft_to_rgb, 
    lucid_colorspace_to_rgb, 
    normalize, 
    cossim, 
    image_buf_to_rgb, 
    show_rgb
)

from .transforms import random_resize

class dreamer():
    def __init__(self, model, progress = True, device = 'cuda'):
        self.model = model 
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.default_func = default_func_mean
        self.transforms = None

    def get_image_param(self, height, width, sd, device):
        image_parameter = init_image_param(height = height, width = width, sd = 0.01, device = device)
        return image_parameter

    def get_optimizer(self, params_list, optimizer = None, lr = 1e-3, weight_decay = 0.):
        if optimizer is not None:
            optimizer = optimizer(params_list, lr = lr, weight_decay = weight_decay)
        else:
            optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_default_transforms(self, rotate, scale_max, scale_min):
        self.transforms= transforms.Compose([
            random_resize(max_size_factor = scale_max, min_size_factor = scale_min),
            transforms.RandomAffine(degrees = rotate)
        ])

    def set_custom_transforms(self, transforms):
        self.transforms = transforms

    def render(self, width, height, iters, layers, lr, rotate_degrees, scale_max = 1.1,  scale_min = 0.5, custom_func = None, weight_decay = 0., grad_clip = 1.):

        image_parameter = self.get_image_param(height = height, width = width, sd = 0.01, device = self.device)
        image_parameter.requires_grad_()

        self.optimizer = self.get_optimizer(
            params_list = [image_parameter],
            lr = lr,
            weight_decay= weight_decay
        )

        if self.transforms is None:
            self.get_default_transforms(rotate = rotate_degrees, scale_max = scale_max, scale_min= scale_min)
        else:
            print("using your custom transforms")


        hooks = []
        for layer in layers:
            hook = Hook(layer)
            hooks.append(hook)

        for i in tqdm(range(iters)):
            self.optimizer.zero_grad()

            img = fft_to_rgb(height, width, image_parameter)
            img = lucid_colorspace_to_rgb(img)
            img = torch.sigmoid(img)
            img = normalize(img)
            img = self.transforms(img)

            if i % 100 ==0:
                import matplotlib.pyplot as plt

                foo = img.detach()[0].cpu().permute(1,2,0)
                plt.imshow(foo)
                plt.show()

            model_out = self.model(img)


            layer_outputs = []

            for hook in hooks:
                out = hook.output[0]
                layer_outputs.append(out)

            if custom_func is not None:
                loss = custom_func(layer_outputs)
            else:
                loss = self.default_func(layer_outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(image_parameter,grad_clip)
            self.optimizer.step()
        

        for hook in hooks:
            hook.close()

        return image_buf_to_rgb(height, width, image_parameter)


# def visualize_feature(model, layer, feature, start_image=None, last_hook_out=None,
#                       size=200, steps=500, lr=0.004, weight_decay=0.1, grad_clip=1,
#                       debug=False, frames=10, show=True, **kwargs):
#     h,w = size if type(size) is tuple else (size,size)
    
    # img_buf = init_fft_buf(h, w, **kwargs)
    # img_buf.requires_grad_()
    # opt = torch.optim.AdamW([img_buf], lr=lr, weight_decay=weight_decay)

    # hook_out = None
    # def callback(m, i, o):
    #     nonlocal hook_out
    #     hook_out = o
    # hook = layer.register_forward_hook(callback)
    
    # for i in range(1,steps+1):
    #     opt.zero_grad()
        
#         img = fft_to_rgb(h, w, img_buf, **kwargs)
#         img = lucid_colorspace_to_rgb(img)
#         img = torch.sigmoid(img)
#         img = normalize(img)
# #         img = lucid_transforms(img, **kwargs)          
#         model(img.cuda())        
#         if feature is None:
#             loss = -1 * hook_out[0].pow(2).mean()
#         else:
#             loss = -1 * hook_out[0][feature].mean()
#         if last_hook_out is not None:
#             simularity = cossim(hook_out[0], last_hook_out, **kwargs)
#             loss = loss + loss * simularity

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(img_buf,grad_clip)
#         opt.step()
        
#         if debug and (i)%(int(steps/frames))==0:
#             clear_output(wait=True)
#             label = f"step: {i} loss: {loss:.2f} stats:{stats}"
#             show_rgb(image_buf_to_rgb(h, w, img_buf, **kwargs),
#                      label=label, **kwargs)

#     hook.remove()
    
#     retval = image_buf_to_rgb(h, w, img_buf, **kwargs)
#     if show:
#         if not debug: show_rgb(retval, **kwargs)
#     return retval, hook_out[0].clone().detach()