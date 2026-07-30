import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from model import CNN

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
  )
])

test_dataset = torchvision.datasets.CIFAR10(
  root = "./data",
  train = False,
  download=True,
  transform= transform
)

test_loader = DataLoader(
  test_dataset,
  batch_size= 32,
  shuffle=True
)


model = CNN()

model.load_state_dict(torch.load("cnn_model.pth"))

model.eval()

correct = 0
total = 0

with torch.no_grad():
  for inputs, labels in test_loader:

    outputs = model(inputs)

    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)

    correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"Validation Accuracy : {accuracy:.2f}%")




 