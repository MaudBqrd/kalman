import numpy as np
import os
from os.path import join
from numpy import loadtxt
import matplotlib.pyplot as plt

# PREPROCESSING FUNCTIONS

def convert_v_to_a(v):
    a = np.gradient(v)
    return a


def simu_to_dataset(path_dataset_simu, path_new_dataset, std_noise):
    list_files = os.listdir(path_dataset_simu)

    if not os.path.exists(join(path_new_dataset, 'train')):
        os.makedirs(join(path_new_dataset, 'train'))

    if not os.path.exists(join(path_new_dataset, 'val')):
        os.makedirs(join(path_new_dataset, 'val'))

    if not os.path.exists(join(path_new_dataset, 'test')):
        os.makedirs(join(path_new_dataset, 'test'))

    compteur_train = 0
    compteur_test = 0
    computeur_val = 0

    for file in list_files:

        data = loadtxt(join(path_dataset_simu, file), delimiter=";")
        data = data.reshape((-1, 2, data.shape[1]))

        for i in range(data.shape[0]):
            v_vehicle = data[i][0]
            v_wheel = data[i][1]
            a_vehicle = convert_v_to_a(v_vehicle)

            # add noise
            a_vehicle += np.random.normal(0, std_noise[0], size=a_vehicle.shape)
            v_wheel += np.random.normal(0, std_noise[1], size=v_wheel.shape)

            to_save = np.zeros((3, len(a_vehicle)))

            if 'TEST' in file:
                save_path = join(path_new_dataset, 'test', f'test_length_{len(a_vehicle)}_seq_{compteur_test}.npy')
                compteur_test += 1
            else:
                if np.random.binomial(1, 0.9, size=1) == 1:
                    save_path = join(path_new_dataset, 'train', f'train_length_{len(a_vehicle)}_seq_{compteur_train}.npy')
                    compteur_train += 1
                else:
                    save_path = join(path_new_dataset, 'val', f'val_length_{len(a_vehicle)}_seq_{computeur_val}.npy')
                    computeur_val += 1

            to_save[0] = a_vehicle
            to_save[1] = v_wheel
            to_save[2] = v_vehicle

            plt.plot(a_vehicle, label='a_vehicle')
            ax = plt.gca()
            ax.set(xlabel="t [s]", ylabel="a [m/s²]")
            plt.legend()
            plt.figure()
            plt.plot(v_wheel, label='v_wheel')
            ax = plt.gca()
            ax.set(xlabel="t [s]", ylabel="v [m/s]")
            plt.legend()

            plt.show()
            exit(0)

            np.save(save_path, to_save)



if __name__ == '__main__':
    # path_to_dataset = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/dataset"
    # path_new_dataset = "/home/nathan/Bureau/Mines/MAREVA/Mini projet/kalman_dataset"

    path_to_dataset = "/home/maud/Documents/mines/mareva/mini_projet/dataset"
    path_new_dataset = "/home/maud/Documents/mines/mareva/mini_projet/kalman_dataset"

    simu_to_dataset(path_to_dataset, path_new_dataset, std_noise=[1,3])