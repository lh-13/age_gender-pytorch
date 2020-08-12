'''
train.py  年龄与性别对应的训练程序
author:lh-13
date:2020.0805
'''
import torch
import numpy as np  
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader      
import torch.optim as optim 
import torch.backends.cudnn as cudnn  
from core.dataset import Dataset  
from core.net import AgeGenderNet
from core.resnet import resnet18, resnet34 
import os   

if __name__ == "__main__":
    cuda = True     #是否使用cuda 
    trainset = Dataset("./data/dataset/UTKFace/cropAlignFace/UTKFace")
    num_dataset = len(trainset)
    dataloader = DataLoader(trainset, batch_size=4, num_workers=0, shuffle=True)
    writer = SummaryWriter(log_dir='log')

    #创建模型
    #model = AgeGenderNet()
    model = resnet18()

    #加载预训练模型
    model_path = ""
    if os.path.exists(model_path):
        print('Loading {} weights into state dict...'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('pretrained load finished !!!!!!')
    else:
        print('not load pretrained!!!!!')


    num_epochs = 300 
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    net = model.train()
    if cuda:
        print("let's use", torch.cuda.device_count(), 'GUPs')
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    #损失函数
    mse_loss = nn.MSELoss()    #均方损失函数
    cross_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = 0
        for index, sample_batched in enumerate(dataloader):
            images_batch, age_batch, gender_batch = sample_batched['image'], sample_batched['age'], sample_batched['gender']
            if cuda: 
                images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()
            optimizer.zero_grad()

            age_pred, gender_pred = model(images_batch)
            age_batch = age_batch.view(-1, 1)
            gender_batch = gender_batch.long()

            #calculate the batch loss 
            loss = mse_loss(age_pred, age_batch) + cross_loss(gender_pred, gender_batch)
            loss.backward()
            optimizer.step()

            #update training loss         
            train_loss += loss.item()
        train_loss = train_loss / num_dataset
        writer.add_scalar("train_loss", train_loss, global_step=epoch)
        print("Epoch:{} \t Training Loss:{:.6f}".format(epoch, train_loss))

        if (epoch % 10 == 0):
            model.eval()
            torch.save(model, 'checkpoint/Epoch%d-train_loss-%.4f.pth'%(epoch, train_loss))

    #save model 
    model.eval() 
    torch.save(model, 'checkpoint/age_gender_model.pt')




    
