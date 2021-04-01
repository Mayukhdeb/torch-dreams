import unittest
from unittest import TestCase

import torchvision.models as models
from torch_dreams.dreamer import dreamer

from torch_dreams.auto_image_param import auto_image_param
from torch_dreams.model_bunch import ModelBunch
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

        dreamy_boi = dreamer(model = model, device= 'cpu', quiet= False)

        image_param = dreamy_boi.render(
            layers = [model.Mixed_6a],
            iters = 5
        )

        image_param.save(filename = 'test_single_model.jpg')

        self.assertTrue(os.path.exists('test_single_model.jpg'))
        self.assertTrue(isinstance(image_param, auto_image_param), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_single_model.jpg')

    def test_custom_size(self):

        model = models.inception_v3(pretrained=True)

        dreamy_boi = dreamer(model = model, device= 'cpu', quiet= False)

        image_param = dreamy_boi.render(
            layers = [model.Mixed_6a],
            iters = 5,
            width = 255,
            height = 255
        )

        image_param.save(filename = 'test_custom_size.jpg')

        self.assertTrue(os.path.exists('test_custom_size.jpg'))
        self.assertTrue(isinstance(image_param, auto_image_param), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')
        os.remove('test_custom_size.jpg')

    def  test_single_model_custom_func(self):
        model = models.inception_v3(pretrained=True)

        dreamy_boi = dreamer(model = model, device= 'cpu', quiet= False)

        image_param = dreamy_boi.render(
            layers = [model.Mixed_6a],
            iters = 5,
            custom_func= make_custom_func(layer_number= 0, channel_number= 10)
        )

        image_param.save(filename = 'test_single_model_custom_func.jpg')
        
        self.assertTrue(os.path.exists('test_single_model_custom_func.jpg'))
        self.assertTrue(isinstance(image_param, auto_image_param), 'should be an instance of auto_image_param')
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

        dreamy_boi = dreamer(model = bunch, quiet= False, device= 'cpu')

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
        self.assertTrue(isinstance(image_param, auto_image_param), 'should be an instance of auto_image_param')
        self.assertTrue(isinstance(image_param.__array__(), np.ndarray))
        self.assertTrue(isinstance(image_param.to_hwc_tensor(), torch.Tensor), 'should be a torch.Tensor')
        self.assertTrue(isinstance(image_param.to_chw_tensor(), torch.Tensor), 'should be a torch.Tensor')

        os.remove('test_multiple_models_custom_func.jpg')


if __name__ == '__main__':

    unittest.main()
