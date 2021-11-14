import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms_train = transforms.Compose([
    # We create random crops and etc
    transforms.RandomResizedCrop(size=(64,64), scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_transforms_val_test = transforms.Compose([
    # We create random crops and etc
    transforms.Resize(72),
    transforms.CenterCrop(64),
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


