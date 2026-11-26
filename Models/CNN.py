import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, input_c=3, output=6):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(input_c, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )
        self.fc = nn.Linear(50 * 9 * 9, output)

    def forward(self, x):
        # Untuk input 100x100, sisi gambar berubah 100 → 96 → 32 → 28 → 9.
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return self.fc(torch.flatten(x, 1))  # Satu logit per kelas.
