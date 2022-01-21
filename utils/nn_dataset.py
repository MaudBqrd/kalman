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


def compute_normalizing_constants_dataset(path_to_dataset, max_samples=1000):
    list_files = os.listdir(path_to_dataset)
    n = min(len(list_files), max_samples)

    mean = np.zeros((3,1))
    std = np.zeros((3,1))

    for i in range(1, len(list_files)):
        file = list_files[i]
        c_data = np.load(join(path_to_dataset, file))
        mean[:,0] += 1/n * np.mean(c_data, axis=1)
        std[:, 0] += 1 / n * np.std(c_data, axis=1)

    return mean, std


if __name__ == '__main__':

    # test of nn_dataset functions

    path_training_data = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset/train"
    mean, std = compute_normalizing_constants_dataset(path_training_data)

    train_dataset = KalmanDataset(path_training_data, mean, std)
    a_vehicle, v_wheel, v_vehicle, idx = train_dataset[0]




