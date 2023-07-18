from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageNet

from random import sample
import cv2
import numpy as np

class GaussianBlur(object):
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target =  self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None(img):
            img_1 = self.transform(img)
            img_2 = self.transform(img)           
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img_1, img_2, target
    
class CIFAR100Pair(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None(img):
            img_1 = self.transform(img)
            img_2 = self.transform(img)            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img_1, img_2, target
    
class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target =  self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if self.transform is not None(img):
            img_1 = self.transform(img)
            img_2 = self.transform(img)
            
        return img_1, img_2, target
    
class ImageNetPair(ImageNet):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None(img):
            img_1 = self.transform(img)
            img_2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img_1, img_2, target

def image_dataset(dataset_name, root='../data', pair=True):
    if dataset_name in ['CIFAR10', 'cifar10']:
        if pair:
            train_data = CIFAR10Pair(root=root, train=True, transforms=train_data, download=True)
            test_data = CIFAR10Pair(root=root, train=True, transforms=test_data)
        else:
            train_data = CIFAR10(root=root, train=True, transforms=train_data)
            test_data = CIFAR10(root=root, train=True, transforms=test_data)
    elif dataset_name in ['CIFAR100', 'cifar100']:
        if pair:
            train_data = CIFAR100Pair(root=root, train=True, transforms=train_data)
            test_data = CIFAR100Pair(root=root, train=True, transforms=test_data)
        else:
            train_data = CIFAR100(root=root, train=True, transforms=train_data)
            test_data = CIFAR100(root=root, train=True, transforms=test_data)
    elif dataset_name in ['STL10', 'stl10']:
        if pair:
            train_data = STL10Pair(root=root, train=True, transforms=train_data)
            test_data = STL10Pair(root=root, train=True, transforms=test_data)
        else:
            train_data = STL10(root=root, train=True, transforms=train_data)
            test_data = STL10(root=root, train=True, transforms=test_data)
    elif dataset_name=='ImageNet':
        train_data = ImageNet(root=root, train=True, transforms=train_data)
        test_data = ImageNet(root=root, train=True, transforms=test_data)
    
    return train_data, test_data