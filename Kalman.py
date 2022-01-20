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


if __name__ == '__main__':

    Niter = 1
    xin = np.array([0, 1])
    Pini = np.identity(2)
    F = np.identity(2)
    B = np.identity(2)
    u = ([1, 1])
    Q = np.identity(2)

    for iter in range(Niter):
        xpred, Ppred = kalmanPred(xin, F, B, u, Pini, Q)
        print(xpred, Ppred)

        z = np.array([1.1, 1.9])
        H = np.identity(2)
        R = 10*np.identity(2)
        xup, Pup = KalmanUpdate(xpred, Ppred, z, H, R)
        print(xup, Pup)