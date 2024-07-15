import sys
sys.path += ['.', './layers/', './utils/']


import torch

class ClaxLeNet5(nn.Module):
    def __init__(self, in_shape: tuple, ou_dim: int) -> None:
        super().__init__()

        if len(in_shape) != 3:
            raise Exception(f"The parameter in_shape must be a tuple of three elements (channels, width, height), found {in_shape}")
        if ou_dim < 1:
            raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")

        c, w, h = in_shape

        c1 = 6
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c1, kernel_size=5, padding=2, stride=1)
        w1 = size_conv_layer(w, kernel_size=5, padding=2, stride=1)
        h1 = size_conv_layer(h, kernel_size=5, padding=2, stride=1)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        w2 = size_conv_layer(w1, kernel_size=2, padding=0, stride=2)
        h2 = size_conv_layer(h1, kernel_size=2, padding=0, stride=2)

        c2 = 16
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=5, stride=1)
        w3 = size_conv_layer(w2, kernel_size=5, padding=0, stride=1)
        h3 = size_conv_layer(h2, kernel_size=5, padding=0, stride=1)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        w4 = size_conv_layer(w3, kernel_size=2, padding=0, stride=2)
        h4 = size_conv_layer(h3, kernel_size=2, padding=0, stride=2)

        self.flatten_size = w4 * h4 * c2
        fc2_size = int(self.flatten_size * 30 / 100)
        # Prima riduco del 30% per evitare aumenti complessità

        self.fc1 = nn.Linear(self.flatten_size, fc2_size)
        fc3_size = int(fc2_size * 30 / 100)  # Riduci ulteriormente la complessità per il terzo fully connected
        self.fc2 = nn.Linear(fc2_size, fc3_size)
        self.fc3 = nn.Linear(fc3_size, ou_dim) #quindi anziche mettere un layer quantum ho messo un layer fc 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max_pool1(self.relu(self.conv1(x)))
        x = self.max_pool2(self.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        out = self.softmax(x)
        return out
