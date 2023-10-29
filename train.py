from dataset import MnistTrain
from constants import DEVICE
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from tqdm import tqdm


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 28*28),
            nn.Linear(28*28, 10)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == "__main__":
    train_dataset = MnistTrain()
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model = LinearModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        for image, label in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label.flatten())
            loss.backward()
            optimizer.step()

    PATH = "./models/my_model.pt"
    torch.save(model, PATH)
