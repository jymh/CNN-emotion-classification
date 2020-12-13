import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from train import pre_process
from train import Net
from train import device
import numpy as np

def compute_saliency_maps(X, y, model):
    model.eval()

    X.requires_grad_(True)
    saliency = None

    logits = model(X)
    #print(logits)
    logits = logits.gather(1, y.view(-1, 1)).squeeze()
    logits.backward(torch.FloatTensor([1., 1., 1., 1.]).to(device))

    saliency = abs(X.grad.data)

    return saliency.squeeze()

def show_saliency_maps(X, y, model):
    saliency = compute_saliency_maps(X, y, model)

    N = X.shape[0]
    X = np.squeeze(X.cpu().detach().numpy())
    y = np.squeeze(y.cpu().detach().numpy())
    saliency = np.squeeze(saliency.cpu().detach().numpy())

    for i in range(N):
        mean = np.mean(saliency[i])
        plt.subplot(2, N, i+1)
        plt.imshow(X[i])
        plt.axis('off')

        plt.subplot(2, N, N+i+1)
        plt.imshow(saliency[i], cmap=plt.cm.hot, vmin=mean)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)

    plt.show()

def select_monotype(datax, datay, type):
    index = []
    for i in range(len(datax)):
        if datay[i] == type:
            index.append(i)
        if len(index) == 4:
            return index


X, y = pre_process("train.json", True)
index = select_monotype(X, y, 3)
X_todraw, y_todraw = X.index_select(0, torch.tensor(index)).to(device), y.index_select(0, torch.tensor(index))
y_todraw = torch.LongTensor(y_todraw).to(device)
#print(y_todraw.view(-1,1))

model = torch.load("resnet_model.pth")
show_saliency_maps(X_todraw, y_todraw, model)
