import os
import sys

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CIFAR10Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "cifar10")
        self.transform = transform
        self.train = train

        if download:
            self.download()

        self.images, self.labels = self.load_data()
        self.num_classes = 10
    
    def download(self):
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=True)
    
    def load_data(self):
        if self.train:
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=False)
        else:
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=False)
        
        images = []
        labels = []
        for image, label in dataset:
            images.append(image)
            labels.append(label)
        
        return images, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "index": index,
            "image": image,
            "label": label,
        }

class CIFAR100Dataset(Dataset):
    def __init__(self, root, transform=None, download=False, train=True):
        self.root = os.path.join(root, "cifar100")
        self.transform = transform
        self.train = train

        if download:
            self.download()
        self.num_classes = 100
        
        self.images, self.labels = self.load_data()
    
    def download(self):
        torchvision.datasets.CIFAR100(root=self.root, train=True, download=True)
        torchvision.datasets.CIFAR100(root=self.root, train=False, download=True)
    
    def load_data(self):
        if self.train:
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=True, download=False)
        else:
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=False, download=False)
        
        images = []
        labels = []
        for image, label in dataset:
            images.append(image)
            labels.append(label)
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "index": index,
            "image": image,
            "label": label,
        }
    
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10 = CIFAR10Dataset(root="data", transform=transform, download=True, train=True)
    cifar100 = CIFAR100Dataset(root="data", transform=transform, download=True, train=True)

    print(len(cifar10))
    print(len(cifar100))
    print(cifar10[0])
    print(cifar100[0])