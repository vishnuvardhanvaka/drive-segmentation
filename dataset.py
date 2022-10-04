import os
import numpy as np
import cv2
import torch
from glob import glob
from torch.utils.data import Dataset,DataLoader
import cv2
import sys

class DriveDataset(Dataset):
    def __init__(self,root,is_train):
        self.root=root
        if is_train:
            dic='/train/'
        else:
            dic='/test/'
        self.image=sorted(glob(root+dic+'/images/*'))
        self.mask=sorted(glob(root+dic+'/masks/*'))

    def __getitem__(self,index):
        im=str(self.image[index])
        ma=str(self.mask[index])
        return im,ma

        return t,index
    def __len__(self):
        return len(self.image)
        

if __name__=='__main__':

    root='new_data'
    dataset=DriveDataset(root,is_train=True)
    loader=DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    print(len(dataset))
    for i,j in enumerate(loader):
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
            cv2.destroyAllWindows()
        

















