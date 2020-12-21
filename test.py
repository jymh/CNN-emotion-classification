import torch
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from train import pre_process
from train import batch_size
from train import Net
from train import device
import numpy as np
import pandas as pd


def write_result(prediction):
    id = [i+1 for i in range(len(prediction))]
    pred_dict = {"ID": id, "emotion": prediction}
    res = pd.DataFrame(pred_dict)
    res.to_csv(r"prediction_results/res.csv", index=None)
    print("Success")


def predict(model, test_dataloader):
    model.eval()

    prediction = []

    for index, pixels in enumerate(test_dataloader):
        pixel = pixels[0].to(device)
        output = model(pixel)
        output_list = output.argmax(dim=1).tolist()
        prediction.extend(output_list)

    return prediction


    return prediction


test_x = pre_process("test.json",False)

model = torch.load("models/final_model.pth")

test_dataset = torch.utils.data.TensorDataset(test_x)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          pin_memory = True)

model = model.to(device)
prediction = predict(model, test_loader)
write_result(prediction)

