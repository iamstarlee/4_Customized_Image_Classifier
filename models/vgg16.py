import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=4):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096) #(25088, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        # print("vgg16_features' size is {}".format(vgg16_features.size()))
        out = vgg16_features.view(out.size(0), -1)
        
        # print("the input size of layer6 is {}".format(out.size()))
        # print("the output size of layer6 is {}".format(self.layer6(out)))
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

      


# if __name__ == '__main__':
#     from PIL import Image
#     import os
#     import numpy as np
#     import torch
#     import sys
    
#     from torch.utils.data import DataLoader, Dataset
#     import glob
#     import torchvision.transforms as transforms
    
#     sys.path.insert(0,"D:\Documents\VS2022Projects\pytorch-cifar100-master")
#     import config
#     from utils import get_custom_dataloader

#     ## 验证集数据路径
#     image_dir = r'D:\Documents\VS2022Projects\pytorch-cifar100-master\data\\'
#     #data_dir = glob.glob(image_dir+'*.*')

#     transform_test = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
#                             std  = [ 0.229, 0.224, 0.225 ]),
#     ])

#     test_loader = get_custom_dataloader(config.test_dir)
#     train_loader = get_custom_dataloader(config.train_dir)

#     from torch import nn
#     import torch.nn.functional as F
#     from models.vgg16 import VGG16

#     path = r'D:\Documents\VS2022Projects\pytorch-cifar100-master\checkpoint\vgg16\Wednesday_27_July_2022_00h_35m_31s\\'
#     state_dict = torch.load(path + 'vgg16-100-regular.pth')
#     model = VGG16(n_classes=4)
#     model.load_state_dict(state_dict)
#     model.eval()

#     for step, (image, label) in enumerate(test_loader.dataset):
#         if(step == 11):
#             print(label)
#             print(image.size())
#             vgg16_features, output = model(image.unsqueeze(0)) #image.unsqueeze(0)
#             print(vgg16_features)
#             _, predict = output.topk(1, 1, largest=True, sorted=True)
#             print(predict)
#             y_pred = torch.argmax((predict),axis=1).numpy()
#             print("the predict label is {}".format(y_pred))
