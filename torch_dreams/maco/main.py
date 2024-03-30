import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.models as models

from torch_dreams.maco.features_visualizations import maco
from torch_dreams.maco.features_visualizations.objectives import Objective
from torch_dreams.maco.plots.image import plot_maco

# %matplotlib inline
import matplotlib.pyplot as plt


resnet50 = models.resnet50(pretrained=True)


classes = [
           (1,    'Goldfish'),
           (33,   'Loggerhead turtle'),
           (75,   'Black widow'),
           (96,   'Toucan'),
           (107,  'Jellyfish'),
           (145,  'Penguin'),
           (213,  'Irish setter'),
           (294,  'Brown Bear'),
           (301,  'Ladybug'),
           (309,  'Bee'),
           (323,  'Monarch Butterfly'),
           (393,  'Anemone fish')
]


for logit_id, class_name in classes:
  # create the objective, '-1' is the last layer, and the c_id's are the ids of the classes
  obj_logits = Objective.layer(resnet50, "conv1", logit_id)

  img, alpha = maco(obj_logits, nb_steps=128, values_range = (-1, 1))

  plot_maco(img, alpha)
  plt.title(class_name)
  plt.show()