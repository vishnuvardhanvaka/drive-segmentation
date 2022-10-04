import os
import numpy as np
import cv2
import imageio
from glob import glob
from tqdm import tqdm
import sys

from albumentations import HorizontalFlip,VerticalFlip,Rotate

#create directory ..
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'dir for path {path} created successfully')
    else:
        print('path already exists')

def load_data(path):
    train_x=sorted(glob(os.path.join(path,'training','images','*.tif')))
    train_y=sorted(glob(os.path.join(path,'training','1st_manual','*.gif')))

    test_x=sorted(glob(os.path.join(path,'test','images','*.tif')))
    test_y=sorted(glob(os.path.join(path,'test','mask','*.gif')))

    return (train_x,train_y),(test_x,test_y)

def augment_data(images,masks,save_path,augment=True):
    size=(512,512)
    for idx,(x,y) in tqdm(enumerate(zip(images,masks)),total=len(images)):
        #extracting name...
        x_name=x.split('\\')[-1].split('.')[0]

        #reading image and mask
        x=cv2.imread(x,1)
        y=imageio.mimread(y)[0]

        if augment==True:
             aug=HorizontalFlip(p=1.0)
             augmented=aug(image=x,mask=y)
             x1=augmented['image']
             y1=augmented['mask']

             aug=VerticalFlip(p=1.0)
             augmented=aug(image=x,mask=y)
             x2=augmented['image']
             y2=augmented['mask']

             aug=Rotate(limit=45,p=1.0)
             augmented=aug(image=x,mask=y)
             x3=augmented['image']
             y3=augmented['mask']

             X=[x,x1,x2,x3]
             Y=[y,y1,y2,y3]
             
            
        else:
            X=[x]
            Y=[y]

        index=0
        for i,m in zip(X,Y):
            i=cv2.resize(i,size)
            m=cv2.resize(m,size)

            tiname=f'{x_name}_{index}.png'
            tmname=f'{x_name}_{index}.png'

            image_path=os.path.join(save_path,'image',tiname)
            mask_path=os.path.join(save_path,'mask',tmname)

            cv2.imwrite(image_path,i)
            cv2.imwrite(mask_path,m)
            index+=1
            
        


if __name__=='__main__':
    #data loading...
    data_path='datasets/retina_vessels'
    (train_x,train_y),(test_x,test_y)=load_data(data_path)

    #create directory to save augumented data..
    create_dir('new_data/train/image/')
    create_dir('new_data/train/mask/')
    create_dir('new_data/test/image/')
    create_dir('new_data/test/mask/')

    'data agumentation'
    augment_data(train_x,train_y,'new_data/train/',augment=True)
    augment_data(test_x,test_y,'new_data/test/',augment=False)












