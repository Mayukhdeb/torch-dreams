# torch-dreams
deep dreams in PyTorch

<code><img width="31%" src="https://github.com/Mayukhdeb/torch-dreams/blob/master/images/torch_dream_tiger_layer_15.gif?raw=true"></code>
<code><img width="31%" src="https://github.com/Mayukhdeb/torch-dreams/blob/master/images/torch_dream_tiger_layer_20.gif?raw=true"></code>
<code><img width="31%" src="https://github.com/Mayukhdeb/torch-dreams/blob/master/images/torch_dream_tiger_layer_27.gif?raw=true"></code>

## Less lines of code, more deep-dreams

```python
from torch_dreams.simple import vgg19_dreamer
import cv2 ## for saving images

simple_dreamer = vgg19_dreamer()

dreamed_image = simple_dreamer.dream(
    image_path = "your_image.png",
    layer_index= 27,
    iterations= 2,
    size = (256,256)
)

cv2.imwrite("dream.jpg", dreamed_image)
```

## deep-dreams on a video

```python
from torch_dreams.simple import vgg19_dreamer
simple_dreamer = vgg19_dreamer()


simple_dreamer.deep_dream_on_video(
    video_path = "sample_videos/tiger_mini.mp4",
    save_name = "dream.mp4",
    layer = simple_dreamer.layers[13],
    octave_scale= 1.3,
    num_octaves = 2,
    iterations= 2,
    lr = 0.09,
    size = None, 
    framerate= 30.0
)

```
## Generating deep dreams with your own PyTorch model

* importing `torch_dreams`
```python
from torch_dreams import  utils
from torch_dreams import dreamer
import matplotlib.pyplot as plt ## for viewing the deep-dreams
```
* choosing a model (could be some other PyTorch model as well)
```python
model= models.vgg19(pretrained=True)
model.eval()
```
* selecting one of the model's layers for the deep-dream

```python
layers = list(model.features.children())
layer = layers[13]
```

* Defining the torch transforms to be applied before the forward pass  (could be any set of torch transforms). Or if you're using the VGG19 like me, you could use `utils.preprocess_func_vgg` and `utils.deprocess_func_vgg`

```python
preprocess = utils.preprocess_func_vgg
deprocess = utils.deprocess_func_vgg
```
* Calling an instance of the `dreamer` class and generating a deep-dream

```python
dreamer = dreamer(model, preprocess, deprocess)

dreamed = dreamer.deep_dream(
                        image_np =image_sample, 
                        layer = layer, 
                        octave_scale = 1.5, 
                        num_octaves = 2, 
                        iterations = 2, 
                        lr = 0.09,
                        )
plt.imshow(dreamed)
plt.show()                        
```
