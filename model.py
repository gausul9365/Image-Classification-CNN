import torch.nn as nn



class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d( 3,16, kernel_size = 3)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    self.conv2 = nn.Conv2d(16, 32, kernel_size= 3)
    
    # classification
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(32*6*6, 128)
    self.fc2 = nn.Linear(128, 10)

    
  def forward(self, x):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)


    x = self.flatten(x)

    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)

    return x


