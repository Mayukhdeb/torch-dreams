A python package to reverse engineer neural nets for interpretability.

## Summary

When a deep-learning model looks for a potted plant, what does it actually look for? Does it look for pots or plants? perhaps both? This is exactly what torch-dreams can help us answer.

<p align="center">
  <img src="banner.png" />    
   <br>A breakdown of a segmentation model’s classes with torch-dreams

</p>

Torch-dreams is a tool that helps reverse engineer CNNs and visualize what each layer/channel/neuron looks for in an input. It aims to make model interpretability more accessible and open for general use.

## Statement of need

With the advent of CNNs into fields like healthcare and developmental biology, it has been more important than ever to understand and visualize the representations learned by the models to justify the decisions taken by it.

Different model architectures are generally compared based on performance metrics like accuracy, loss and speed. The common element in these approaches is that they tell us “how good a model is” but not “how a model thinks”.

Torch-dreams helps us gain an insight on “how a model thinks”. This is done by running optimization algorithms on the input image to maximize the activations of various elements(layers/channels/units) within the CNN. It relies heavily on PyTorch and NumPy.

The aim of this library was inspired by the optvis module in tensorflow/lucid and the the Feature Visualization paper, but torch-dreams has been re-written from scratch to give the user complete freedom over selection of layers/channels/units and determining how exactly should the activations be optimized.

## Usage

Torch-dreams can be used to maximize the activations of certain features within a CNN by “training” an image towards an objective.

In order to reduce high frequency patterns, it uses various regularization techniques such as scaling, random jitter, random rotations, etc. All of which can be adjusted by the user as per required.

One of the key advantages of this library is the amount of freedom that it provides. The user can write their own custom objective functions and use a custom set of torchvision transforms to fully customize how the optimization happens at each iteration

A good example is the ability to simultaneously optimize features from different models in a single image parameter. This would open up the possibility to visualize features from ensemble models.


<p align="center">
  <img src="modelbunch.jpeg" />    
   <br> Maximizing the activations of features from 2 different models
</p>

Another important area of research that can be explored are adversarial attacks. The user can break down what each layer “saw” in an adversarial image using caricatures. This opens up the possibility of interpreting how adversarial examples affect the model’s predictions on a micro scale.


<p align="center">
  <img src="adversarial.jpeg" />    
   <br> Caricatures reveal how each layer “sees” the adversarial example.
</p>


Torch-dreams is simple enough for artists to generate fascinating patterns, at the same time its also flexible enough for researchers to explore new horizons in the domain of interpretability.

## Acknowledgements

- Gene Kogan for his valuable feedback + insight through the genesis of this project.
- The friendly folks over on distill slack

## References

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). <https://doi.org/10.1038/s41586-020-2649-2>

Olah, et al., "Feature Visualization", Distill, 2017. [https://distill.pub/2017/feature-visualization/ ](https://distill.pub/2017/feature-visualization/)<https://github.com/tensorflow/lucid>

<https://github.com/tensorflow/lucid/issues/121>
