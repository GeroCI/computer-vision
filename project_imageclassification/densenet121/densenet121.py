# coding=utf-8
from __future__ import print_function, division
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data import *
import matplotlib.pyplot as plt
from model import resnet34
from torchvision import transforms, models
# from models import *
#from my_model import resnet34_wlcat
import time
import os
import copy
plt.ion()

# 加载数据集

#f1 = open('/home/xingyue/xiejia/Record_loss_acc_train.txt','w')
#f2 = open('/home/xingyue/xiejia/Record_loss_acc_val.txt','w')
def train(model,criterion,optimizer,scheduer,dataloader,num_epochs):

    since = time.time()
    best_acc = 0.0
    x_epoch = range(num_epochs)
    yt_loss = []
    yt_acc = []
    yv_loss = []
    yv_acc = []
    t = []
    weight1 = []
    weight2 = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        print('*'*20)
        for phase in ['train', 'val']:
            # 开始训练
            if phase == 'train':
                scheduer.step()#测试就不用更新权重
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0
            dataset_sizes = 0.0
       
            for i,data in enumerate(dataloader[phase]):
                inputs,labels = data
                # inputs, lables = Variable(inputs), Variable(lables)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs.data,1)
                    # 优化和训练过程
                    loss = criterion(outputs,labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                #统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                dataset_sizes += labels.size(0)

                # t.append(i + epoch * 600)
                # weight1.append(model.w1.cpu())
                # weight2.append(model.w2.cpu() )
            # print(model.w1)
            # plt.figure(num=1, figsize=(8, 5))
            # plt.plot(t, weight2, 'r.', markersize=5)
            # plt.plot(t, weight1, 'b.', markersize=5)
            # plt.show()
            # print(dataset_sizes)
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.float() / dataset_sizes
            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if phase =='train':
               yt_loss.append(epoch_loss)
               yt_acc.append(epoch_acc)
               #f1.write(str(epoch_loss))
              # f1.write(str(epoch_acc))
            if phase =='val':
               yv_loss.append(epoch_loss)
               yv_acc.append(epoch_acc)  
               #f2.write(str(epoch_loss))
               #f2.write(str(epoch_acc))
            if phase=='val'and epoch_acc > best_acc:
               best_acc = epoch_acc
               #torch.save(model, 'models_34_1024_1.pkl')
               best_model_wts = copy.deepcopy(model.state_dict())
               #model.cuda()
               #torch.save(best_model_wts,'model_pkl/%d-%fbest_resnet34_ty.pkl'%(epoch,epoch_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    plt.plot(x_epoch,yt_loss,color='#F08080')
    plt.plot(x_epoch,yt_acc,color='#DB7093')
    plt.xlabel('epoch')
    plt.ylabel('train_loss_acc')
    plt.savefig('./train_loss_acc.jpg')
    plt.show() 
    plt.close()

    plt.plot(x_epoch,yv_loss,color='#F08080')
    plt.plot(x_epoch,yv_acc,color='#DB7093')
    plt.xlabel('epoch')
    plt.ylabel('val_loss_acc')
    plt.savefig('./val_loss_acc.jpg')
    plt.show() 
    plt.close()
   # f1.close()
   # f2.close()
    # 加载最佳模型的权重
    torch.save(best_model_wts,'/home/xingyue/xiejia/model_pkl/best_densenet121_zz.pkl')
    # model.load_state_dict(best_model_wts)
    # save_model(model,'params.pkl')
    return #model




if __name__ == "__main__":
    dataloader = images(4)

    # 加载训练参数并且开始训练模型
    #model_ft = models.resnet34(pretrained=True)
    model_ft = torchvision.models.densenet121(pretrained=True)
    # model_ft =resnet50(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2)
    print(model_ft)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    model_ft = model_ft.to(device)

    train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloader, num_epochs=35)
