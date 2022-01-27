import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.kalman_filter_utils import KalmanFilter1D
from utils.nn_dataset import KalmanDataset, compute_normalizing_constants_dataset
from utils.kalman_networks import NeuralNetwork, NeuralNetwork
import torch.nn as nn
import matplotlib.pyplot as plt

def test(test_dataloader, model, loss_fn):
    """
    Test function for network which sees future
    """
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


def test_realtime(test_dataloader, model, loss_fn, tronc_value, visu=False):
    """
    Test function for realtime network
    """
    print("Test \n")
    lossVal = []

    for batch, X in enumerate(test_dataloader):

        print(batch, len(test_dataloader))

        a_vehicle, v_wheel, v_vehicle, idx = X
        L = a_vehicle.shape[1]

        r_hat = []

        for t in range(L):
            if t + 1 >= tronc_value:
                a_vehicle_trunc = a_vehicle[:, t + 1 - tronc_value:t+1]
                v_wheel_trunc = v_wheel[:, t + 1 - tronc_value:t + 1]
                # v_vehicle_trunc = v_vehicle[:, t + 1 - tronc_value:t + 1]
            else:
                nb_zeros = tronc_value - t - 1
                zeros_add = torch.zeros((len(a_vehicle), nb_zeros))
                a_vehicle_trunc = torch.cat((zeros_add, a_vehicle[:, 0:t+1]), dim=1).to(torch.float32)
                v_wheel_trunc = torch.cat((zeros_add, v_wheel[:, 0:t+1]), dim=1).to(torch.float32)
                # v_vehicle_trunc = np.concatenate((zeros_add, v_vehicle[:, 0:t + 1]), axis=1)

            # Compute prediction error
            r = model(a_vehicle_trunc, v_wheel_trunc)
            r_hat.append(r[:, 0, -1])

        r_hat = torch.reshape(torch.stack(r_hat), (batch_size, 1, -1))

        a_vehicle = a_vehicle * std[0, 0] + mean[0, 0]
        v_wheel = v_wheel * std[1, 0] + mean[1, 0]
        pred, P_hat_tab = KalmanFilter1D(a_vehicle, v_wheel, r_hat, Q_tab=1, ret_P_hat=True, v_init=v_wheel[:,0])

        if visu:
            plt.plot((v_vehicle * std[2, 0] + mean[2, 0])[0])
            plt.plot(pred[0].detach().numpy() )
            plt.figure()
            plt.plot(r_hat[0, 0].detach().numpy() )
            plt.plot((v_wheel - (v_vehicle * std[2, 0] + mean[2, 0]))[0])
            plt.figure()
            plt.plot(P_hat_tab[0].detach().numpy())
            plt.show()
            exit("Process exited. Set visu to False to test on whole trajectories ")

        loss = loss_fn(pred, v_vehicle*std[2,0] + mean[2,0])

        lossVal.append(loss.item())

    loss = np.mean(np.array(lossVal))
    print(f"Test loss: {loss:>7f}")


if __name__ == '__main__':

    # PARAMS
    batch_size = 1
    model_path = "checkpoints/model_deep_realtime.pth"

    # LOAD DATASET

    # path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
    path_training_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/train"
    mean, std = compute_normalizing_constants_dataset(path_training_data)

    path_test_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/test_sample"
    # path_test_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/test"
    test_dataset = KalmanDataset(path_test_data, mean, std)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # TEST FCT

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    loss_fn = nn.MSELoss()

    # test for network which sees future
    # test(test_dataloader, model, loss_fn)

    # test realtime
    test_realtime(test_dataloader, model, loss_fn, tronc_value=30, visu=False)
