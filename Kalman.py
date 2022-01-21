import matplotlib.pyplot as plt
import numpy as np

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

def KalmanPred1D(x_hat, F, B, u, P_hat, Q):
    x_hat = F*x_hat + B*u
    P_hat = F*P_hat*F + Q
    return x_hat, P_hat

def KalmanUpdate1D(x_hat, P_hat, z, H, R):
    y = z - H*x_hat
    S = H*P_hat*H + R
    K = P_hat * H / S
    x_hat = x_hat + K*y
    P_hat = (1 - K*H)*P_hat
    return x_hat, P_hat


def KalmanFilter1D(a_vehicle, v_wheel, R_tab, Q_tab):

    v_hat = 0
    P_hat = 0
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
    plt.plot(x1, v_vehicle, label="v vehicle")
    plt.plot(x1, v_update_hat_tab, label = "v hat (after update)")
    plt.legend()

    if plot_all:

        plt.figure()
        plt.plot(x1, v_wheel, label="v wheel")
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



if __name__ == '__main__':

    path_to_traj = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/train/train_length_490_seq_0.npy"

    data = np.load(path_to_traj)
    a_vehicle = data[0]
    v_wheel = data[1]
    v_vehicle = data[2]

    # R_tab = np.array(len(a_vehicle)*[10])
    R_tab = 2*np.abs(a_vehicle)
    Q_tab = np.array(len(a_vehicle)*[0.25])

    v_predict_hat_tab, v_update_hat_tab, P_hat_tab = KalmanFilter1D(a_vehicle, v_wheel, R_tab, Q_tab)
    KalmanVisualisation(v_predict_hat_tab, v_update_hat_tab, P_hat_tab, v_vehicle, v_wheel, a_vehicle, plot_all=False)
    mse = eval_kalman(v_update_hat_tab, v_vehicle)

    plt.show()