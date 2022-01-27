import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.kalman_networks import NeuralNetwork
from utils.nn_dataset import KalmanDataset, KalmanDatasetTronque, compute_normalizing_constants_dataset
import os
from os.path import join
from utils.kalman_filter_utils import KalmanFilter1D


def weighted_mse_loss(input, target):
    weight = torch.ones_like(input)
    weight[:,-1] = 30
    return torch.mean(weight * (input - target) ** 2)


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer, mean, std):
    """
    Training function for realtime time AND future network
    """
    model.train()

    loss_train = []
    for batch, X in enumerate(train_dataloader):
        a_vehicle, v_wheel, v_vehicle, idx = X

        # Compute prediction error
        r = model(a_vehicle, v_wheel)

        a_vehicle = a_vehicle * std[0, 0] + mean[0, 0]
        v_wheel = v_wheel * std[1,0] + mean[1,0]
        pred = KalmanFilter1D(a_vehicle[:,-r.shape[-1]:], v_wheel[:,-r.shape[-1]:], r, Q_tab=1,
                              v_init=std[2,0]*v_vehicle[:,-r.shape[-1]] + mean[2,0])

        loss = loss_fn(pred, v_vehicle[:,-r.shape[-1]:]*std[2,0] + mean[2,0])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train.append(loss.item())

    loss_mean_train = np.mean(np.array(loss_train))
    print(f"loss: {loss_mean_train}\n")

    model.eval()

    print("Validation")

    lossVal = []

    for batch, X in enumerate(val_dataloader):
        a_vehicle, v_wheel, v_vehicle, idx = X

        # Compute prediction error
        r = model(a_vehicle, v_wheel)

        a_vehicle = a_vehicle * std[0, 0] + mean[0, 0]
        v_wheel = v_wheel * std[1, 0] + mean[1, 0]
        pred = KalmanFilter1D(a_vehicle[:, -r.shape[-1]:], v_wheel[:, -r.shape[-1]:], r, Q_tab=1,
                              v_init=std[2, 0] * v_vehicle[:, -r.shape[-1]] + mean[2, 0])

        loss = loss_fn(pred, v_vehicle[:, -r.shape[-1]:] * std[2, 0] + mean[2, 0])

        lossVal.append(loss.item())

    loss_mean_val = np.mean(np.array(lossVal))
    print(f"Validation loss: {loss_mean_val:>7f}")

    return loss_mean_train, loss_mean_val


if __name__ == '__main__':

    # HYPERPARAMETERS

    epochs = 150
    batch_size = 64
    truncate_size = 30
    lr = 1e-3
    weight_decay = 1e-4
    model_name = "model_deep_realtime"

    # LOAD DATASET

    # path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
    path_training_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/train"
    mean, std = compute_normalizing_constants_dataset(path_training_data)

    train_dataset = KalmanDatasetTronque(path_training_data, mean, std, truncate_size)  # for realtime time network
    # train_dataset = KalmanDataset(path_training_data, mean, std)  # for a network which can see the future

    path_val_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/val"
    # path_val_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/val"
    # val_dataset = KalmanDataset(path_val_data, mean, std)
    val_dataset = KalmanDatasetTronque(path_val_data, mean, std, truncate_size)

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = NeuralNetwork()
    # print(model)

    # loss_fn = nn.MSELoss()  # future
    loss_fn = weighted_mse_loss  # realtime
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        loss_train, loss_val = train(train_dataloader, val_dataloader, model, loss_fn, optimizer, mean ,std)
        if len(val_loss) > 0 and loss_val < np.min(val_loss):
            torch.save(model.state_dict(), join("checkpoints", f"{model_name}.pth"))
            print(f"Saved PyTorch Model State to checkpoints/{model_name}.pth")
        train_loss.append(loss_train)
        val_loss.append(loss_val)

    print("Done!")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    plt.plot(np.array(train_loss), label="train loss")
    plt.plot(np.array(val_loss), label="val loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()
