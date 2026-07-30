import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transform
from model import CNN
from torchvision import transforms



transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
  )
])

train_dataset  = torchvision.datasets.CIFAR10(
  root="./data",
  train= True,
  download=True,
  transform=transform
)

train_loader = DataLoader(
  train_dataset,
  batch_size=32,
  shuffle=True
)

test_dataset = torchvision.datasets.CIFAR10(
  root = "./data",
  train = False,
  download= True,
  transform= transform
  )

test_loader = DataLoader(
  test_dataset,
  batch_size= 32,
  shuffle= False
)



model = CNN()

critrion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
  model.parameters(),
  lr = 0.001
)

for epoch in range(10):

  model.train()

  running_loss = 0

  for inputs, labels in train_loader:
  
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = critrion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(
    f"Epoch : {epoch + 1} / {10}"
    f"Loss: {running_loss / len(train_loader):.4f}"
    )
  


# validation

model.eval()

correct = 0
total = 0

with torch.no_grad():

  for images, labels in test_loader:

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    # it retuns two things
    # The maximum values (the highest raw scores).
    # The indices where those maximum values occurred. 
    # _, The underscore is a Python convention used to throw away values you don't need.
    # therefore it only store the predicted class indices (0 to 9) in predicted (a tensor of shape [32]

    correct += (predicted == labels).sum().item()
    total += labels.size(0)

accuracy = 100 * correct / total 

print(f"Validation Accuracy : {accuracy:.2f}%")



# Save model 

torch.save(model.state_dict(), "cnn_model.pth")
print("Model saved successfully!")
