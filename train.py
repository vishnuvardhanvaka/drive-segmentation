import os
import time
from glob import glob
import cv2
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from model import UNET
from loss import Loss
from utils import create_dir,epoch_time

from dataset import DriveDataset


if __name__=='__main__':

    'Directories'
    create_dir('files')

    'hyperparameters'
    h=512
    w=512
    size=(h,w)
    batch_size=1
    epochs=20
    lr=1e-4
    checkpoint='files/checkpoint.pth'

    #transforms...
    transform=transforms.Compose([
        transforms.ToTensor(),

        ])
    def cvt_pil(image):
        trans=transforms.ToPILImage()
        return trans(image)

    #load dataset...l
    
    a=Image.open('eye.png')
    a=transform(a)
    a=a.view(1,3,512,512)
    mask=Image.open('mask.png')
    mask=transform(mask)
    mask=mask.view(1,1,512,512)
    nmask=nn.Sigmoid()(mask)
    print(a.shape,mask.shape)

    train_dataset=DriveDataset('new_data',is_train=True)
    test_dataset=DriveDataset('new_data',is_train=False)

    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    

    model=UNET()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.BCELoss()
    if os.path.exists('seg2.pth'):
        load=torch.load('seg2.pth')
        model.load_state_dict(load['model_state'])
        optimizer.load_state_dict(load['optim_state'])

    
    y_pred=model(a)
    img=cvt_pil(y_pred.view(1,512,512))
    img.show() 

    for epoch in range(epochs):

        y_pred=model(a)
        pred=nn.Sigmoid()(y_pred)
        loss_fn=Loss()
        loss=loss_fn(pred,mask)
        
        print(f'loss of {loss} at epoch {epoch+1}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1)%5==0:
            d={
                'epoch':5,
                'model_state':model.state_dict(),
                'optim_state':optimizer.state_dict()
                }
            torch.save(d,'seg2.pth')
            img=cvt_pil(y_pred.view(1,512,512))
            img.show()
            
        
  

    '''
    for i,j in enumerate(train_loader):
        image=j[0][0]
        mask=j[1][0]
        img=cv2.imread(image)
        ma=cv2.imread(mask)
        sim=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
        sma=cv2.resize(ma,(0,0),fx=0.5,fy=0.5)
        canvas=np.zeros((256,512,3),np.uint8)
        canvas[:256,:256]=sim
        canvas[:256,256:]=sma
        cv2.imshow('image',canvas)
        
        if cv2.waitKey(1)==ord('q'):
            sys.exit()
        else:
            cv2.waitKey(1000)
            cv2.destroyAllWindows()'''




















