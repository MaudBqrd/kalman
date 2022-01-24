import numpy as np
from torch.utils.data import DataLoader
from utils.kalman_filter_utils import KalmanFilter1D
from utils.nn_dataset import KalmanDataset, compute_normalizing_constants_dataset
from utils.kalman_networks import NeuralNetwork
import torch.nn as nn


def test(test_dataloader, model, loss_fn):
    print("Test \n")
    lossVal = []

    for batch, X in enumerate(test_dataloader):
        a_vehicle, v_wheel, v_vehicle, idx = X

        # Compute prediction error
        r = model(a_vehicle, v_wheel)

        a_vehicle = a_vehicle * std[0, 0] + mean[0, 0]
        v_wheel = v_wheel * std[1, 0] + mean[1, 0]
        pred = KalmanFilter1D(a_vehicle, v_wheel, r, Q_tab=1)

        loss = loss_fn(pred, v_vehicle*std[2,0] + mean[2,0])

        lossVal.append(loss.item())

    loss = np.mean(np.array(lossVal))
    print(f"Test loss: {loss:>7f}")


if __name__ == '__main__':

    # PARAMS
    batch_size = 64
    model_path = "checkpoints/model.pth"

    # LOAD DATASET

    # path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
    path_training_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/train"
    mean, std = compute_normalizing_constants_dataset(path_training_data)

    path_test_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/test"
    # path_test_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/test"
    test_dataset = KalmanDataset(path_test_data, mean, std)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # TEST FCT

    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    test(test_dataloader, model, loss_fn)
