import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from train import pre_process


unloader = transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(0.001)

def save_image(tensor, num):
    dir = 'source_images'
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    path = os.path.join(dir, 'train_image_{}.jpg'.format(num))
    image.save(path)

x, y = pre_process("train.json", True)
for i in range(100):
    save_image(x[i],i)