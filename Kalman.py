import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.kalman_filter_utils import KalmanPred1D, KalmanUpdate1D

def kalmanPred(xin, F, B, u, Pini, Q):

    xpred = np.dot(F, xin) + np.dot(B,u)
    Ppred = np.dot(np.dot(F, Pini), np.transpose(F)) + Q

    return xpred, Ppred

def KalmanUpdate(xpred, Ppred, z, H, R):

    y = z - np.dot(H, xpred)
    S = np.dot(np.dot(H, Ppred), np.transpose(H)) + R
    K = np.dot(np.dot(Ppred, np.transpose(H)), np.linalg.inv(S))
    xup = xpred + np.dot(K, y)
    Pup = np.dot((np.identity(len(K))-np.dot(K,H)),Ppred)

    return xup, Pup


def KalmanFilter1D(a_vehicle, v_wheel, R_tab, Q_tab):
    """
    Numpy implementation of Kalman filter
    """

    v_hat = v_wheel[0]
    P_hat = 1
    t_final = len(a_vehicle)

    v_predict_hat_tab = np.zeros(len(a_vehicle))
    v_update_hat_tab = np.zeros(len(a_vehicle))
    P_hat_tab = np.zeros(2*len(a_vehicle))

    for t in range(t_final):
        v_hat, P_hat = KalmanPred1D(v_hat, F=1, B=1, u=a_vehicle[t], P_hat=P_hat, Q=Q_tab[t])
        v_predict_hat_tab[t] = v_hat
        P_hat_tab[2*t] = P_hat
        v_hat, P_hat = KalmanUpdate1D(v_hat, P_hat, z=v_wheel[t], H=1, R=R_tab[t])
        v_update_hat_tab[t] = v_hat
        P_hat_tab[2 * t + 1] = P_hat

    return v_predict_hat_tab, v_update_hat_tab, P_hat_tab


def KalmanVisualisation(v_predict_hat_tab, v_update_hat_tab, P_hat_tab, v_vehicle, v_wheel, a_vehicle, plot_all=True):

    x1 = np.arange(len(v_vehicle))
    x2 = np.linspace(0,len(P_hat_tab)/2, num=len(P_hat_tab))
    plt.plot(x1, v_wheel, '--', label="v wheel", color="green")
    plt.plot(x1, v_vehicle, label="v vehicle")
    plt.plot(x1, v_update_hat_tab, label = "v hat")
    ax = plt.gca()
    ax.set(xlabel="t [s]", ylabel="v [m/s]")
    plt.legend()

    if plot_all:

        plt.figure()
        plt.plot(x1, v_wheel, '--', label="v wheel", color="green")
        plt.plot(x1, v_update_hat_tab, label="v hat (after update)")
        plt.legend()

        plt.figure()
        plt.plot(x1, v_vehicle, label="v vehicle")
        plt.plot(x1, v_predict_hat_tab, label='v hat prediction')
        plt.plot(x1, v_update_hat_tab, label='v hat update')
        plt.legend()

        plt.figure()
        plt.plot(x2, P_hat_tab, label="P hat")
        plt.legend()


def eval_kalman(v_update_hat_tab, v_vehicle):
    mse = np.mean(np.square(v_update_hat_tab - v_vehicle))
    print(f"MSE: {mse}")
    return mse


def test_classique(dataset, loss_fn, R_tab, Q_tab):
    """
    Compute the mean MSE for the classical Kalman filter on the test set
    """
    print("Test \n")
    lossVal = []

    for i in range(len(dataset)):
        X = dataset[i]
        a_vehicle, v_wheel, v_vehicle, idx = X

        # Compute prediction error
        v_predict_hat_tab, v_update_hat_tab, P_hat_tab = KalmanFilter1D(a_vehicle, v_wheel, R_tab, Q_tab)

        loss = loss_fn(torch.from_numpy(v_update_hat_tab), v_vehicle)

        lossVal.append(loss.item())

    loss = np.mean(np.array(lossVal))
    print(f"Test loss: {loss:>7f}")



if __name__ == '__main__':

    # UNCOMMENT TO GET THE MSE ON THE WHOLE TEST SET FOR CLASSICAL KALMAN FILTER
    """path_test_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/test_sample"
    # path_test_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/test"
    # path_test_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/test"
    test_dataset = KalmanDataset(path_test_data, np.zeros((3,1)), np.ones((3,1)))

    R_tab = np.array(490 * [200])
    Q_tab = np.array(490 * [1])
    test_classique(test_dataset, nn.MSELoss(), R_tab, Q_tab)"""


    # UNCOMMENT TO APPLY CLASSICAL KALMAN FILTER ON ONE TRAJECTORY (path_to_traj)
    path_to_traj = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/test_sample/test_length_490_seq_6.npy"

    data = np.load(path_to_traj)
    a_vehicle = data[0]
    v_wheel = data[1]
    v_vehicle = data[2]

    R_tab = np.array(len(a_vehicle)*[150])
    # R_tab = 1.5*np.abs(a_vehicle)
    Q_tab = np.array(len(a_vehicle)*[1])

    v_predict_hat_tab, v_update_hat_tab, P_hat_tab = KalmanFilter1D(a_vehicle, v_wheel, R_tab, Q_tab)
    KalmanVisualisation(v_predict_hat_tab, v_update_hat_tab, P_hat_tab, v_vehicle, v_wheel, a_vehicle, plot_all=False)
    mse = eval_kalman(v_update_hat_tab, v_vehicle)

    plt.show()