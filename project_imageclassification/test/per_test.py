# coding=utf-8
from __future__ import print_function

from model import *

import model

import time

import torch.nn as nn

import os

from data_test import *

from PIL import Image

import torch


import torchvision

from torchvision import transforms

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = '/home/xingyue/xiejia/model_pkl/best_resnet34_cc.pkl'     # path to the ckpt file

testdata_dir='/home/xingyue/xiejia/large/'#'/home/xingyue/xiejia/test/data_test_zz/val/1'

# load model

resNet34 = model.resnet34(pretrained=True)
num_ftrs = resNet34.fc.in_features
resNet34.fc = nn.Linear(num_ftrs, 2)
resNet34.cuda()         
resNet34.load_state_dict(torch.load(ckpt))
resNet34.eval()

# Define transformation
composed = transforms.Compose([
             transforms.Resize(224),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

f = open('/home/xingyue/xiejia/test/error/small_large/small_small.txt','w')#写文件前声明权限 
count = [0,0]
prevent = time.time()
for i in range(2):
    current_dir=testdata_dir+str(i)
    img_list = os.listdir(current_dir)
    if i==0:
       f.write('无裂纹图片中预测错误的图片名字:\n')
    if i==1:
       f.write('有裂纹图片中预测错误的图片名字:\n')
    for img_name in img_list: 
       
        print('Processing image: ' + img_name)

        img = Image.open(os.path.join(current_dir, img_name))

        img = composed(img)
     
        img = img.unsqueeze(0)

        img = img.to(device)

        output = resNet34(img)

        _,prediction = torch.max(output.data,1)
  
        prediction = prediction.cpu()
    
        prediction = prediction.numpy().astype(int)

        labels = prediction[0]

        #if labels==i:

           #f.write(img_name+'\n')

           #img_show = Image.open(os.path.join(current_dir, img_name))
   
   
           #img_show.save(os.path.join('/home/xingyue/xiejia/test/right/small_large/'+str(i),img_name))


        if labels!=i:

           count[i]+=1

           f.write(img_name+'\n')

           img_show = Image.open(os.path.join(current_dir, img_name))
   
   
           img_show.save(os.path.join('/home/xingyue/xiejia/test/error/small_large/'+str(i),img_name))

sum = 1190-(count[0]+count[1])

time_sum =time.time() - prevent

time_average = float(time_sum)/1190

acc = float(sum)/1190*100

f.write('总错误图片数是:'+str(1190-sum)+'\n')

f.write('精度: '+str(acc)+'%\n')

f.write('总耗时: '+str(time_sum)+'ms\n')

f.write('每张图片平均耗时: '+str(time_average)+'ms\n')

f.close()



