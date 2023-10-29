import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import LinearModel
from dataset import MnistTest
from tqdm import tqdm


model = torch.load("./models/my_model.pt")
model.eval()
test_loader = DataLoader(MnistTest(), batch_size=4)

total_loss = 0.0
correct = 0
total_samples = 0

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for image, label in tqdm(test_loader):
        outputs = model(image)
        loss = criterion(outputs, label.flatten())  # Make them both 1d
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += label.size(0)

        correct += (predicted == label.T).sum().item()

print(f"total_loss: {total_loss}")
print(f"correct: {correct}")
print(f"total_samples: {total_samples}")
