import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 20 

class Net(nn.Module):
    def __init__(self, input_size=64):
        super(Net, self).__init__()
        if input_size == 64:
            self.fc1_input = 320
        else:
            self.fc1_input = 11520
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(self.fc1_input, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, self.fc1_input)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
