""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
from xml.etree.ElementInclude import default_loader

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataloader import generate_datasets
import pandas as pd
from torchvision import transforms
import config
from PIL import Image
import matplotlib.pyplot as plt

from models.vgg import VGG

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net =='vggpy':
        from models.vgg16 import VGG16
        net = VGG16(config.NUM_CLASSES)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader2():
    
    Customdata_Train, datasets_count1, Customdata_Test, datasets_count2 = generate_datasets()
    return Customdata_Train, datasets_count1, Customdata_Test, datasets_count2

def get_custom_dataloader(root):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
   
    train_dataset = custom_dataloader(root = root,  loader = default_loader, transform=transform_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    return train_loader
#定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]   
    #print("Before sorting, the classes are {}".format(classes))
    # 遍历dir目录下的所有子目录名称并将其存在classes中
    #classes.sort()
    #print("After sorting, the classes are {}".format(classes.sort()))
    # 由于Python版本的不同可能需要更换为sorted(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    #print("class_to_index is {}".format(class_to_idx))
    # 创建一个字典，将类别与数字对应
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)   		# 把path中包含的"~"和"~user"转换成用户目录
    for target in sorted(os.listdir(dir)):  # os.listdir()函数返回一个包含dir目录下所有文件或目录的列表
        d = os.path.join(dir, target)       # 将dir和target连接形成新的路径  d为类别目录
        
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            #print("root is {} and fnames is {}\n".format(root, fnames))
            for fname in sorted(fnames):
                # if has_file_allowed_extension(fname, extensions):   # 判断fnames的后缀是否正确（JPEG.JPG等等）
                path = os.path.join(root, fname)                # 得到文件的路径和文件名
                item = (path, class_to_idx[target])		        # 依据class_to_idx得到图片类别对应的数字
                images.append(item)
    # for idx, (path1, idx1) in enumerate(images):
    #     print("the path is {} and the index is {}".format(path1, idx1))
    # path, target = images[1]
    # print("path is {} and target is {}".format(path, target))
    return images

class custom_dataloader(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader
        # self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        
        sample = self.loader(path)
        
        # target(label) transformer code is not finished, but I'm going to do it in the future. 
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, torch.tensor(target) #return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        return len(self.samples)






# The Cifar dataloader code is commented 
# def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
#     """ return training dataloader
#     Args:
#         mean: mean of cifar100 training dataset
#         std: std of cifar100 training dataset
#         path: path to cifar100 training python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: train_data_loader:torch dataloader object
#     """

#     transform_train = transforms.Compose([
#         #transforms.ToPILImage(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     #cifar100_training = CIFAR100Train(path, transform=transform_train)
#     cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     cifar100_training_loader = DataLoader(
#         cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    

#     return cifar100_training_loader

# def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
#     """ return training dataloader
#     Args:
#         mean: mean of cifar100 test dataset
#         std: std of cifar100 test dataset
#         path: path to cifar100 test python dataset
#         batch_size: dataloader batchsize
#         num_workers: dataloader num_works
#         shuffle: whether to shuffle
#     Returns: cifar100_test_loader:torch dataloader object
#     """

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#     #cifar100_test = CIFAR100Test(path, transform=transform_test)
#     cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#     cifar100_test_loader = DataLoader(
#         cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

#     return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


if __name__ == '__main__':
    path = r'D:\Documents\VS2022Projects\pytorch-cifar100-master\data\test'
    Custom_test_loader = get_custom_dataloader(path)
    # for step, (image, label) in enumerate(Custom_train_loader):
    #     print("In step {}, the label is {} and the image is {}".format(step, label, image))
    # visualization code 
    # for step ,(b_x,b_y) in enumerate(Custom_train_loader):
    #     if step < 3:
    #         imgs = torchvision.utils.make_grid(b_x)
    #         imgs = np.transpose(imgs,(1,2,0))
    #         plt.imshow(imgs)
    #         plt.show()