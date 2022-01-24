import torch

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


def KalmanFilter1D(a_vehicle, v_wheel, r, Q_tab):
    batch_size = a_vehicle.shape[0]
    v_hat = torch.zeros(batch_size)
    P_hat = torch.zeros(batch_size)
    t_final = a_vehicle.shape[1]

    v_update_hat_tab = torch.zeros((batch_size,a_vehicle.shape[1]))

    for t in range(t_final):
        v_hat, P_hat = KalmanPred1D(v_hat, F=1, B=1, u=a_vehicle[:,t], P_hat=P_hat, Q= Q_tab)
        v_hat, P_hat = KalmanUpdate1D(v_hat, P_hat, z=v_wheel[:,t], H=1, R=r[:, 0, t])
        v_update_hat_tab[:,t] = v_hat

    return v_update_hat_tab