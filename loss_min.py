import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.kalman_filter_utils import KalmanFilter1D
from utils.nn_dataset import KalmanDataset, compute_normalizing_constants_dataset
from utils.kalman_networks import NeuralNetwork
import torch.nn as nn
import matplotlib.pyplot as plt


def calcul_l_min(train_dataloader, model, tronc_value):
    print("Test \n")
    lmin = []

    for batch, X in enumerate(train_dataloader):
        a_vehicle, v_wheel, v_vehicle, idx = X
        current_data = torch.stack([a_vehicle, v_wheel, v_vehicle])
        L = a_vehicle.shape[1]
        temps_fin = np.random.randint(L+1)

        if temps_fin >= tronc_value :
            data_tronc = current_data[:, :, temps_fin-tronc_value:temps_fin]
        else :
            nb_zeros = tronc_value - temps_fin
            zeros_add = torch.zeros((len(current_data), a_vehicle.shape[0], nb_zeros))
            data_tronc = torch.cat((zeros_add, current_data[:, :, 0:temps_fin]), dim =2)

        a_vehicle_trunc, v_wheel_trunc, v_vehicle_trunc = data_tronc[0], data_tronc[1], data_tronc[2]


        r = model(a_vehicle_trunc, v_wheel_trunc)


        a_vehicle_trunc = a_vehicle_trunc * std[0, 0] +  mean[0, 0]
        v_wheel_trunc = v_wheel_trunc * std[1, 0] + mean[1, 0]
        v_vehicle_trunc = v_vehicle_trunc * std[2, 0] + mean[2, 0]

        pred, P_hat_tab = KalmanFilter1D(a_vehicle_trunc, v_wheel_trunc, r, 1, ret_P_hat=True, v_init = v_vehicle_trunc[:, 0])
        #plt.plot(v_vehicle_trunc[0])
        #plt.plot(pred[0].detach().numpy())
        #plt.show()

        lmin.append(torch.mean(P_hat_tab).item())
    return np.mean(lmin)


if __name__ == '__main__':
    # TO GET A BOUND on the training loss

    # PARAMS
    batch_size = 64
    model_path = "checkpoints/model_deep_realtime.pth"

    # LOAD DATASET

    # path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
    path_training_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/train"
    mean, std = compute_normalizing_constants_dataset(path_training_data)

    train_dataset = KalmanDataset(path_training_data, mean, std)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # TEST FCT

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))

    loss_fn = nn.MSELoss()
    # test(test_dataloader, model, loss_fn)

    mean_tab = []

    for i in range(100):
        mean_tab.append(calcul_l_min(train_dataloader, model, tronc_value=30))

    print("l_min : ", np.mean(mean_tab))
