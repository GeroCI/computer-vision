# coding=utf-8
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
def images(batch_size):
    data_transform = {
        'train':transforms.Compose([
         transforms.Resize(256),transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]),
        'val': transforms.Compose([
             transforms.Resize(224),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
]),
    }

    # 加载数据
    images_path = '/home/xingyue/xiejia/data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(images_path, x),
                                              data_transform[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #class_names = image_datasets['train'].classes
    return dataloaders
if __name__ == "__main__":
    train = images(8)
    # print(len(train))
    for i, data in enumerate(train['val']):
        imgs,labels = data
        print(imgs.size(0))
        imgs = torchvision.utils.make_grid(imgs,nrow = 4)
        imshow(imgs)
        print(names)
        plt.axis('off')
        plt.show()
