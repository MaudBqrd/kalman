import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from utils.nn_dataset import KalmanDataset, compute_normalizing_constants_dataset
import os
from os.path import join
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt



path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
mean, std = compute_normalizing_constants_dataset(path_training_data)

path_test_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/test"
test_dataset = KalmanDataset(path_test_data, mean, std)

batch_size = 1

# Create data loaders.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = NeuralNetwork(Q=1, mean = mean, sd = std)
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()

for  _, (a_vehicle, v_wheel, v_vehicle, idx) in enumerate(test_dataloader):
    with torch.no_grad():
        pred, r = model(a_vehicle, v_wheel)
        print(pred.shape)
        # print(v_vehicle*std[2,0] + mean[2,0].shape)
        plt.plot((v_vehicle*std[2,0] + mean[2,0])[0])
        plt.plot(pred[0])
        plt.figure()
        plt.plot(r[0, 0])
        plt.show()
    break
