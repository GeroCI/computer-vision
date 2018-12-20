# coding=utf-8
import model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from data_test import *

def test(model,criterion,dataloader):

    since = time.time()
    best_acc = 0.0

    t = []
    weight1 = []
    weight2 = []
    model.train(False)
    running_loss = 0.0
    running_corrects = 0.0
    dataset_sizes = 0.0

    for i,data in enumerate(dataloader['val']):
        inputs,labels = data
        # inputs, lables = Variable(inputs), Variable(lables)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        loss = criterion(outputs,labels)
        #统计
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        dataset_sizes += labels.size(0)
	print(str(i) + ' : ' + str(running_corrects))

    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.float() / dataset_sizes
    print('{} Loss:{:.4f} Acc:{:.4f}'.format('val',epoch_loss,epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

dataloader = images(1)
resNet34 = model.resnet34(pretrained=True)
num_ftrs = resNet34.fc.in_features
resNet34.fc = nn.Linear(num_ftrs, 2)
resNet34.cuda()

resNet34.load_state_dict(torch.load('model_pkl/best_resnet34_zz.pkl'))

criterion = nn.CrossEntropyLoss()
# optimizer_ft = optim.SGD(resNet34.parameters(),lr= 0.01,momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=15,gamma=0.1)

test(resNet34,criterion,dataloader)


