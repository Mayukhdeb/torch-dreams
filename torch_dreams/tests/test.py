import unittest
from unittest import TestCase

import torchvision.models as models
import torchvision.transforms as transforms
from torch_dreams import Dreamer

from torch_dreams.auto_image_param import AutoImageParam
from torch_dreams.custom_image_param import CustomImageParam
from torch_dreams.masked_image_param import MaskedImageParam
from torch_dreams.transforms import random_resize   
from torch_dreams.model_bunch import ModelBunch
from torch_dreams.image_transforms import InverseTransform

import numpy as np
import torch
import os 

    
def make_custom_func(layer_number = 0, channel_number= 0): 
    def custom_func(layer_outputs):
        loss = layer_outputs[layer_number][channel_number].mean()
        return -loss
    return custom_func

class test(unittest.TestCase):

    def test_single_model(self):

        model = models.inception_v3(pretrained=True)

        dreamy_boi = Dreamer(model = model, device= 'cpu', quiet= False)

        image_param = dreamy_boi.render(
            layers = [model.Mixed_6a],
            iters = 5
        )

        image_param.save(filename = 'test_single_model.jpg')

        self.assertTrue(os.path.exists('test_single_model.jpg'))
        self.assertTrue(isinstance(image_param, AutoImageParam), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_single_model.jpg')

    def test_custom_size(self):

        model = models.inception_v3(pretrained=True)

        dreamy_boi = Dreamer(model = model, device= 'cpu', quiet= False)

        image_param = dreamy_boi.render(
            layers = [model.Mixed_6a],
            iters = 5,
            width = 255,
            height = 255
        )

        image_param.save(filename = 'test_custom_size.jpg')

        self.assertTrue(os.path.exists('test_custom_size.jpg'))
        self.assertTrue(isinstance(image_param, AutoImageParam), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_custom_size.jpg')

    def  test_single_model_custom_func(self):
        model = models.inception_v3(pretrained=True)

        dreamy_boi = Dreamer(model = model, device= 'cpu', quiet= False)

        image_param = dreamy_boi.render(
            layers = [model.Mixed_6a],
            iters = 5,
            custom_func= make_custom_func(layer_number= 0, channel_number= 10)
        )

        image_param.save(filename = 'test_single_model_custom_func.jpg')
        
        self.assertTrue(os.path.exists('test_single_model_custom_func.jpg'))
        self.assertTrue(isinstance(image_param, AutoImageParam), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_single_model_custom_func.jpg')

    def test_multiple_models_custom_func(self):

        model1  = models.inception_v3(pretrained=True).eval()
        model2  = models.resnet18(pretrained= True).eval() 

        bunch = ModelBunch(
            model_dict = {
                'inception': model1,
                'resnet':  model2
            }
        )

        layers_to_use = [
            bunch.model_dict['inception'].Mixed_6a,
            bunch.model_dict['resnet'].layer2[0].conv1
        ]

        dreamy_boi = Dreamer(model = bunch, quiet= False, device= 'cpu')

        def custom_func(layer_outputs):
            loss =  layer_outputs[1][89].mean() + layer_outputs[0].mean()**2
            return -loss

        image_param = dreamy_boi.render(
            layers = layers_to_use,
            custom_func= custom_func,
            iters= 5
        )

        image_param.save(filename = 'test_multiple_models_custom_func.jpg')

        self.assertTrue(os.path.exists('test_multiple_models_custom_func.jpg'))
        self.assertTrue(isinstance(image_param, AutoImageParam), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')

        os.remove('test_multiple_models_custom_func.jpg')

    def test_custom_image_param(self):

        model = models.inception_v3(pretrained=True)

        dreamy_boi = Dreamer(model = model, device= 'cpu', quiet= False)
        param = CustomImageParam(image = 'images/sample_small.jpg', device= 'cpu')

        image_param = dreamy_boi.render(
            image_parameter= param,
            layers = [model.Mixed_6a],
            iters = 5,
            lr = 2e-4,
            grad_clip = 0.1,
            weight_decay= 1e-1
        )

        image_param.save(filename = 'test_custom_image_param.jpg')

        self.assertTrue(os.path.exists('test_custom_image_param.jpg'))
        self.assertTrue(isinstance(image_param, CustomImageParam), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_custom_image_param.jpg')

    def test_custom_image_param_set_param(self):
        """
        checks if custom_image_param.set_param correctly 
        loads the image without discrepancies with an absolute tolerance of 1e-5 element-wise
        """
        model = models.inception_v3(pretrained=True)

        dreamy_boi = Dreamer(model = model, device= 'cpu', quiet= False)
        param = CustomImageParam(image = 'images/sample_small.jpg', device= 'cpu')

        image_param = dreamy_boi.render(
            image_parameter= param,
            layers = [model.Mixed_6a],
            iters = 5,
            lr = 2e-4,
            grad_clip = 0.1,
            weight_decay= 1e-1
        )

        image_tensor = image_param.to_nchw_tensor()

        image_param.set_param(tensor = image_tensor)
        # print(torch.abs((image_tensor - image_param.to_nchw_tensor())).mean())

        self.assertTrue(torch.allclose(image_tensor ,image_param.to_nchw_tensor(), atol = 1e-5))

    def test_MaskedImageParam(self):

        model = models.inception_v3(pretrained=True)

        dreamy_boi = Dreamer(model = model, device= 'cpu', quiet= False)

        mask_tensor = torch.ones(1,3,512,512)
        mask_tensor[:,:,:256,:] = 0.

        param = MaskedImageParam(
            image = 'images/sample_small.jpg',
            mask_tensor = mask_tensor,
            device = 'cpu'
        )

        image_param = dreamy_boi.render(
            image_parameter= param,
            layers = [model.Mixed_6a],
            iters = 5,
            lr = 2e-4,
            grad_clip = 0.1,
            weight_decay= 1e-1
        )

        image_param.save(filename = 'test_MaskedImageParam.jpg')

        self.assertTrue(os.path.exists('test_MaskedImageParam.jpg'))
        self.assertTrue(isinstance(image_param, CustomImageParam), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_MaskedImageParam.jpg')     

    def test_caricature(self):

        model = models.resnet18(pretrained=True)

        dreamy_boi = Dreamer(model, device = 'cpu')
        param = CustomImageParam(image = 'images/sample_small.jpg', device= 'cpu')

        image_tensor = param.to_nchw_tensor()

        param = dreamy_boi.caricature(
            input_tensor = image_tensor, 
            layers = [model.layer3],
            power= 1.0,
            iters = 5
        )

        self.assertTrue(isinstance(param, AutoImageParam), 'should be an auto_image_param')

    def test_static_caricature(self):

        model = models.resnet18(pretrained=True)

        dreamy_boi = Dreamer(model, device = 'cpu')
        param = CustomImageParam(image = 'images/sample_small.jpg', device= 'cpu')

        image_tensor = param.to_nchw_tensor()

        param = dreamy_boi.caricature(
            input_tensor = image_tensor, 
            layers = [model.layer3],
            power= 1.0,
            iters = 5,
            static= True
        )

        self.assertTrue(isinstance(param, AutoImageParam), 'should be an auto_image_param')

    def test_custom_normalization(self):

        model = models.resnet18(pretrained=True)

        dreamy_boi = Dreamer(model, device = 'cpu')

        t = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        dreamy_boi.set_custom_normalization(normalization_transform = t)

        param = dreamy_boi.render(
            layers = [model.layer3],
            iters = 5
        )

        expected_transforms = transforms.Compose([

            transforms.RandomAffine(15, translate= (0,0)),
            random_resize(max_size_factor = 1.2, min_size_factor = 0.5),
            InverseTransform(
                old_mean = [0.485, 0.456, 0.406],
                old_std = [0.229, 0.224, 0.225],
                new_transforms = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            )
        ])

        self.assertTrue(expected_transforms , dreamy_boi.transforms)

if __name__ == '__main__':

    unittest.main()
