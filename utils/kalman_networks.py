import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size= 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size= 5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size = 5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size = 3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )

    def forward(self, a_vehicle, v_wheel):
        x = torch.cat((torch.unsqueeze(a_vehicle, dim=1), torch.unsqueeze(v_wheel, dim=1)), dim=1)
        r = self.conv(x)
        return r