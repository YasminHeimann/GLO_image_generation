import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class GeneratorForMnistGLO(nn.Module):
    """
    very simple generator for MNIST.
    digit image size is 1, 28, 28
    code_dim is the number of units in code (it's a 1-dimensional vector)
    taken from here    https://github.com/tneumann/minimal_glo/blob/master/glo.py
    """
    def __init__(self, code_dim, out_channels=1):
        super(GeneratorForMnistGLO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, 84), nn.ReLU(True),
            nn.Linear(84, 120), nn.ReLU(True),
            nn.Linear(120, 16*5*5), nn.ReLU(True),
            View(shape=(-1, 16, 5, 5)),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 16, 5),
            nn.BatchNorm2d(16), nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, out_channels, 5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, code):
        return self.net(code)

    def test(self):
        code_dim = 50
        batch_size = 32
        random_tensor = torch.rand(batch_size, code_dim)
        print(f'the shape of the code is {random_tensor.shape}')
        result = self.forward(random_tensor)
        print(f'the shape of the result is {result.shape}')

