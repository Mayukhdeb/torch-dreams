import torch.nn as nn 

class ModelBunch():
    def __init__(self, model_dict: dict):
        self.model_dict = model_dict 

    def forward(self,x):

        outs = []

        for model in list(self.model_dict.values()):
            output = model(x)
            outs.append(output)

        return outs

    def __call__(self, x):
        return self.forward(x)

    def eval(self):

        for model in list(self.model_dict.values()):
            model.eval()

    def to(self, device):
        for model in list(self.model_dict.values()):
            model.to(device)