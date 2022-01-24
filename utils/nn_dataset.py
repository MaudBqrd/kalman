import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import numpy as np
import os
from os.path import join


class KalmanDataset(data.Dataset):
    """
    torch dataset for training on Kalman filter
    """

    def __init__(self, path_to_dataset, norm_mean, norm_std):
        self.list_data = []
        list_files = os.listdir(path_to_dataset)

        self.length = len(list_files)

        for i in range(len(list_files)):
            self.list_data.append(join(path_to_dataset, list_files[i]))

        self.dataset_type = path_to_dataset.split('/')[-1]
        self.mean = norm_mean
        self.std = norm_std

    def __getitem__(self, index):
        current_data = np.load(self.list_data[index])
        current_data = (current_data - self.mean) / self.std
        a_vehicle, v_wheel, v_vehicle = current_data[0], current_data[1], current_data[2]

        return torch.from_numpy(a_vehicle.astype(np.float32)), torch.from_numpy(v_wheel.astype(np.float32)), \
               torch.from_numpy(v_vehicle.astype(np.float32)), index

    def __len__(self):
        return self.length


class KalmanDatasetTronque(data.Dataset):
    """
    torch dataset for training on Kalman filter
    """

    def __init__(self, path_to_dataset, norm_mean, norm_std, troncature):
        self.list_data = []
        list_files = os.listdir(path_to_dataset)

        self.length = len(list_files)

        for i in range(len(list_files)):
            self.list_data.append(join(path_to_dataset, list_files[i]))

        self.dataset_type = path_to_dataset.split('/')[-1]
        self.mean = norm_mean
        self.std = norm_std
        self.tronc = troncature

    def __getitem__(self, index):
        current_data = np.load(self.list_data[index])
        taille_temps = len(current_data[0])
        temps_fin = np.random.randint(taille_temps+1)

        if temps_fin >= self.tronc :
            data_tronc = current_data[:, temps_fin-self.tronc:temps_fin]
        else :
            nb_zeros = self.tronc - temps_fin
            zeros_add = np.zeros((len(current_data), nb_zeros))
            data_tronc = np.concatenate((zeros_add, current_data[:, 0:temps_fin]), axis =1)
            print(temps_fin, nb_zeros)

        data_tronc = (data_tronc - self.mean) / self.std
        a_vehicle, v_wheel, v_vehicle = data_tronc[0], data_tronc[1], data_tronc[2]

        return torch.from_numpy(a_vehicle.astype(np.float32)), torch.from_numpy(v_wheel.astype(np.float32)), \
               torch.from_numpy(v_vehicle.astype(np.float32)), index

    def __len__(self):
        return self.length

def compute_normalizing_constants_dataset(path_to_dataset, max_samples=1000):
    list_files = os.listdir(path_to_dataset)
    n = min(len(list_files), max_samples)

    mean = np.zeros((3,1))
    std = np.zeros((3,1))

    for i in range(1, len(list_files)):
        file = list_files[i]
        c_data = np.load(join(path_to_dataset, file))
        mean[:,0] += 1/n * np.mean(c_data, axis=1)
        std[:, 0] += 1 /(n-1) * np.std(c_data, axis=1)

    return mean, std


if __name__ == '__main__':

    # test of nn_dataset functions

    path_training_data = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset/train"
    mean, std = compute_normalizing_constants_dataset(path_training_data)

    train_dataset = KalmanDataset(path_training_data, mean, std)
    a_vehicle, v_wheel, v_vehicle, idx = train_dataset[0]




