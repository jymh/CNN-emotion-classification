import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from train import pre_process
import numpy as np
from train import device
from train import Net


unloader = transforms.ToPILImage()

''' draw source images'''
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

def draw_source_images():
    x, y = pre_process("train.json", True)
    for i in range(100):
        save_image(x[i],i)


def plot_training_curve():
    training_acc = [34.07310201088425, 46.18086379250453, 50.125439036628194, 52.85422054112471, 54.811069512524604, 56.30089930140105, 57.451078775715004, 58.342660851441586, 58.91003126326759,
                59.465822687097145, 60.12582500289475, 61.067582693272605, 61.453548959820914, 62.27179744490332, 63.47987185919951, 64.6647882975028, 65.87672237446446, 66.72970782353622,
                67.27777992203482, 68.60936354162646, 69.35813809873017, 70.30761511443899, 71.04481068354626, 72.0599019645683, 72.60411440040141, 73.85078544135243, 74.78096414373384,
                75.13219344629279, 75.86552935273457, 76.57956694584894, 77.03886680304142, 77.475008684241, 77.91501022810607, 78.61360917055849, 79.3392257516693, 79.87185919950596,
                79.89501717549886, 80.35817669535683, 81.46589988035046, 81.4350225790266, 82.23783241344707, 82.87467675325176, 82.69713227063956, 83.0715195491914, 83.49222277972905,
                83.97468061291443, 83.91678567293218, 84.2564359874947, 84.38766451812111, 85.17117603921417, 84.75047280867652, 85.12871974989386, 85.30240456984059, 85.5725809564244,
                86.0974950789301, 85.8466170056737, 86.22872360955652, 86.44100505615809, 86.95819985333281, 86.79995368404802, 86.51819830946775, 87.6915357597746, 86.96977884132927,
                87.11258635995213, 87.57960554247559, 87.51785093982785, 87.82276429040103, 88.01188776100969, 87.64907947045428, 87.96557180902388]
    vali_acc = [33.92857142857143, 43.82142857142857, 47.32142857142857, 49.642857142857146, 53.642857142857146, 52.78571428571428, 52.714285714285715, 53.67857142857143, 52.89285714285714,
            52.46428571428572, 53.607142857142854, 56.53571428571429, 51.964285714285715, 56.85714285714286, 56.57142857142857, 57.85714285714286, 56.107142857142854, 57.07142857142857,
            54.64285714285714, 57.03571428571429, 59.75, 57.53571428571429, 60.14285714285714, 58.785714285714285, 59.10714285714286, 60.71428571428571, 58.53571428571429, 57.785714285714285,
            59.5, 60.60714285714286, 59.64285714285714, 59.57142857142858, 60.57142857142858, 60.10714285714286, 57.785714285714285, 59.07142857142858, 59.392857142857146, 58.60714285714286,
            60.10714285714286, 59.57142857142858, 59.64285714285714, 59.75, 59.67857142857142, 59.0, 59.392857142857146, 59.82142857142857, 60.10714285714286, 59.285714285714285, 59.964285714285715,
            59.32142857142857, 60.03571428571428, 59.10714285714286, 60.21428571428571, 59.82142857142857, 59.60714285714286, 59.53571428571428, 58.214285714285715, 59.32142857142857, 59.0,
            60.357142857142854, 59.67857142857142, 58.17857142857142, 59.82142857142857, 59.67857142857142, 60.17857142857142, 59.32142857142857, 59.64285714285714, 59.21428571428572, 60.0, 60.0]
    training_acc = np.array(training_acc)
    vali_acc = np.array(vali_acc)
    x = np.arange(1, 71, 1)

    fig1 = plt.figure()
    plt.plot(x, training_acc, c='r', label='training accuracy')
    plt.plot(x, vali_acc, c='b', label = 'validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def select_monotype(datax, datay, type, num):
    index = []
    for i in range(len(datax)):
        if datay[i] == type:
            index.append(i)
        if len(index) == num:
            return index

def Plot_filter_result(image_tensor, model, img_type):
    selected_layer1 = model.conv1
    selected_layer2 = model.conv2
    selected_layer3 = model.conv3
    feature = selected_layer1(image_tensor)
    feature = selected_layer2(feature)
    feature = selected_layer3(feature)
    #print(feature.shape)

    path = "filter_output/"

    fig = plt.figure()
    for i in range(16):
        ax = fig.add_subplot(4, 4, i+1)
        filter = feature[:,i,:,:].view(feature.shape[2], feature.shape[3])
        #print(filter.shape)

        filter = filter.cpu().data.numpy()
        filter = 1.0/(1+np.exp(-1*filter))
        filter = np.round(filter*255)
        #print(filter[0])

        ax.imshow(filter)
        plt.axis('off')

    path = path + "Block3Type" + str(img_type) +".png"
    plt.savefig(path)
    #plt.show()


x,y = pre_process("train.json", True)
vali_x = x[:2800]
vali_y = y[:2800]
model = torch.load("models/final_model.pth")
model = model.to(device)
for i in range(7):
    index = select_monotype(vali_x, vali_y, i, 1)
    image_tensor = x.index_select(0, torch.tensor(index)).to(device)
    Plot_filter_result(image_tensor, model, i)


