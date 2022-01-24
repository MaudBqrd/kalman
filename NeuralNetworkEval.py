import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from utils.nn_dataset import KalmanDataset, KalmanDatasetTronque, compute_normalizing_constants_dataset
import os
from os.path import join
from utils.kalman_networks import NeuralNetwork
from utils.kalman_filter_utils import KalmanFilter1D
import matplotlib.pyplot as plt



path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
mean, std = compute_normalizing_constants_dataset(path_training_data)

path_test_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/test"
test_dataset = KalmanDatasetTronque(path_test_data, mean, std, 30)

batch_size = 1

# Create data loaders.
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = NeuralNetwork()
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()

for  _, (a_vehicle, v_wheel, v_vehicle, idx) in enumerate(test_dataloader):
    with torch.no_grad():
        r = model(a_vehicle, v_wheel)
        a_vehicle = a_vehicle * std[0, 0] + mean[0, 0]
        v_wheel = v_wheel * std[1,0] + mean[1,0]
        pred = KalmanFilter1D(a_vehicle, v_wheel, r, 1)
        plt.plot((v_vehicle*std[2,0] + mean[2,0])[0])
        plt.plot(pred[0])
        plt.figure()
        plt.plot(r[0, 0])
        plt.show()
    break
