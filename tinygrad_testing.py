import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD
from dataset import MnistTrain


class LinearModel:
    dims = 28
    n = 100

    def __init__(self):
        self.input = nn.Linear(self.dims*self.dims, self.n*self.n)
        self.output = nn.Linear(self.n*self.n, 10)

    def forward(self, x):
        x = self.input(x.flatten())
        return self.output(x)

    def __call__(self, *args: type, **kwds: type):
        return self.forward(args[0])


if __name__ == "__main__":
    mnist_train = MnistTrain()
    train_images, train_lables = mnist_train.images, mnist_train.labels

    model = LinearModel()

    opt = SGD([model.input.weight, model.output.weight], lr=1e-4)
    print(model(a).numpy())
