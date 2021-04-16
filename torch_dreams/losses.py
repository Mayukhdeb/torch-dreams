import torch.nn as nn
import torch

from .image_transforms import resize_4d_tensor_by_size

class CaricatureLoss(nn.Module):

    def __init__(self, power = 1.):
        super().__init__()        
        self.power = power 

    def cosine_dissimilarity(self,x,y, eps = 1e-6):
        """
        tried my best to replicate: 
            https://github.com/tensorflow/lucid/blob/6dcc927e4ff4e7ef4d9c54d27b0352849dadd1bb/lucid/recipes/caricature.py#L21
            
        if I missed something out, please get in touch with me on Distill slack: @Mayukh 

        or email me: 
            mayukhmainak2000@gmail.com
        or find me on github: 
            github.com/mayukhdeb
        """

        if x.shape != y.shape:
            """
            if their shapes are not equal (likely due to using static caricatures), then resize the target accordingly
            """
            y = resize_4d_tensor_by_size(y.unsqueeze(0), height = x.shape[-2], width = x.shape[-1] ).squeeze(0)

        y = y.detach()
        
        numerator = (x*y.detach()).sum() 
        denominator = torch.sqrt((y**2).sum()) + eps
        cossim = numerator/denominator
        cossim = torch.maximum(torch.tensor(0.1).to(cossim.device), cossim)
        loss = -(cossim*numerator**self.power)
        return loss

    def loss(self, x,y):
        loss = self.cosine_dissimilarity(x,y)  
        return loss

    def forward(self,layer_outputs, ideal_layer_outputs):
        assert len(layer_outputs) == len(ideal_layer_outputs)
        loss = 0.
        for i  in range(len(layer_outputs)):
        
            l  = self.loss(layer_outputs[i] ,ideal_layer_outputs[i])
            loss += l
           
        # print(loss)
        return loss 