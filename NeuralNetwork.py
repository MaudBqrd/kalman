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


path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
mean, std = compute_normalizing_constants_dataset(path_training_data)

train_dataset = KalmanDataset(path_training_data, mean, std)
a_vehicle, v_wheel, v_vehicle, idx = train_dataset[0]


path_test_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/test"
test_dataset = KalmanDataset(path_test_data, mean, std)

path_val_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/val"
val_dataset = KalmanDataset(path_val_data, mean, std)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, Q, mean, sd):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size= 5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size= 5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size = 5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size = 5, padding=2, padding_mode='reflect'),
            nn.ReLU()
        )
        self.Q_tab = Q
        self.mean = mean
        self.sd = sd


    def KalmanPred1D(self, x_hat, F, B, u, P_hat, Q):
        x_hat = F * x_hat + B * u
        P_hat = F * P_hat * F + Q
        return x_hat, P_hat

    def KalmanUpdate1D(self, x_hat, P_hat, z, H, R):
        y = z - H * x_hat
        S = H * P_hat * H + R
        K = P_hat * H / S
        x_hat = x_hat + K * y
        P_hat = (1 - K * H) * P_hat
        return x_hat, P_hat

    def KalmanFilter1D(self, a_vehicle, v_wheel, R_tab):
        batch_size = a_vehicle.shape[0]
        v_hat = torch.zeros(batch_size)
        P_hat = torch.zeros(batch_size)
        t_final = a_vehicle.shape[1]

        v_update_hat_tab = torch.zeros((batch_size,a_vehicle.shape[1]))

        for t in range(t_final):
            v_hat, P_hat = self.KalmanPred1D(v_hat, F=1, B=1, u=a_vehicle[:,t], P_hat=P_hat, Q= self.Q_tab)
            v_hat, P_hat = self.KalmanUpdate1D(v_hat, P_hat, z=v_wheel[:,t], H=1, R=R_tab[:,0,t])
            v_update_hat_tab[:,t] = v_hat

        return v_update_hat_tab

    def forward(self, a_vehicle, v_wheel):
        x = torch.cat((torch.unsqueeze(a_vehicle, dim=1), torch.unsqueeze(v_wheel, dim=1)), dim=1)
        r = self.conv(x)
        a_vehicle = a_vehicle * self.sd[0,0] + self.mean[0,0]
        v_wheel = v_wheel * self.sd[1,0] + self.mean[1,0]
        y = self.KalmanFilter1D(a_vehicle, v_wheel, r)
        return y, r

model = NeuralNetwork(Q=1, mean = mean, sd =std)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    model.train()

    loss_train = []
    for batch, X in enumerate(train_dataloader):
        a_vehicle, v_wheel, v_vehicle, idx = X

        # Compute prediction error
        pred,r = model(a_vehicle, v_wheel)
        loss = loss_fn(pred, v_vehicle*std[2,0] + mean[2,0])


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
        pred, r = model(a_vehicle, v_wheel)
        loss = loss_fn(pred, v_vehicle*std[2,0] + mean[2,0])

        lossVal.append(loss.item())


    loss= np.mean(np.array(lossVal))
    print(f"Validation loss: {loss:>7f}")

def test(test_dataloader, model, loss_fn):
    print("Test \n")
    lossVal = []

    for batch, X in enumerate(test_dataloader):
        a_vehicle, v_wheel, v_vehicle, idx = X

        # Compute prediction error
        pred, r = model(a_vehicle, v_wheel)
        loss = loss_fn(pred, v_vehicle*std[2,0] + mean[2,0])

        lossVal.append(loss.item())

    loss = np.mean(np.array(lossVal))
    print(f"Test loss: {loss:>7f}")





if __name__ == '__main__':
    epochs =30
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train(train_dataloader, val_dataloader, model, loss_fn, optimizer)
    print("Done!")

    test(test_dataloader, model, loss_fn)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    torch.save(model.state_dict(), join("checkpoints","model.pth"))
    print("Saved PyTorch Model State to model.pth")
