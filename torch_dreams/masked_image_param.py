import cv2
import torch
from .custom_image_param import custom_image_param

class masked_image_param(custom_image_param):
    def __init__(self, image, mask, device):
        super().__init__(image = image, device= device)

        self.mask = mask.to(self.device)

        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)/255.
            image = torch.tensor(image).permute(-1,0,1).unsqueeze(0)

        self.original_nchw_image_tensor = image.to(device)

        assert self.mask.shape[-2:] == self.to_nchw_tensor(device = self.device).shape[-2:]

    def to_chw_tensor(self, device = 'cpu'):
        t = self.forward(device= device).squeeze(0).clamp(0,1).detach()  + self.original_nchw_image_tensor.to(device) * (1-self.mask.to(device)) 
        return t.squeeze(0)

    def forward(self, device):
        return self.normalize(self.postprocess(device = device), device= device).clamp(0,1) * self.mask.to(device) 
