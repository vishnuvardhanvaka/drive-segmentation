import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
from PIL import Image

#model...

class conv_block(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.conv1=nn.Conv2d(inc,outc,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(outc)

        self.conv2=nn.Conv2d(outc,outc,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(outc)

        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        return x
class encoder(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.conv=conv_block(inc,outc)
        self.pool=nn.MaxPool2d((2,2))
    def forward(self,x):
        r=self.conv(x)
        x=self.pool(r)
        return r,x
class decoder(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.up=nn.ConvTranspose2d(inc,outc,kernel_size=2,stride=2,padding=0)
        self.conv=conv_block(inc,outc)
    def forward(self,x,skip):
        x=self.up(x)
        x=torch.cat([x,skip],dim=1)
        x=self.conv(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        '...encoder...'
        self.e1=encoder(3,64)
        self.e2=encoder(64,128)
        self.e3=encoder(128,256)
        self.e4=encoder(256,512)

        '...bridge...'
        self.b=conv_block(512,1024)

        '...decoder...'
        self.d1=decoder(1024,512)
        self.d2=decoder(512,256)
        self.d3=decoder(256,128)
        self.d4=decoder(128,64)

        '...output...'
        self.out=nn.Conv2d(64,1,kernel_size=1,padding=0)
    def forward(self,x):
        af=nn.Sigmoid()
        s1,p1=self.e1(x)
        s2,p2=self.e2(p1)
        s3,p3=self.e3(p2)
        s4,p4=self.e4(p3)
        bg=self.b(p4)
        #print(s1.shape,s2.shape,s3.shape,s4.shape,bg.shape)

        dc1=self.d1(bg,s4)
        dc2=self.d2(dc1,s3)
        dc3=self.d3(dc2,s2)
        dc4=self.d4(dc3,s1)
        #print(dc1.shape,dc2.shape,dc3.shape,dc4.shape)

        output=self.out(dc4)
        return output

if __name__=='__main__':
    a=torch.randn(1,3,512,512)
    print(a.shape)
    obj=UNET()
    r=obj(a)
    print(r.shape)
    
    
