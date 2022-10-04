import torch
import torch.nn as nn
import torchvision


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_pred,mask):
            
            
            pred=y_pred.view(-1)
            mask=mask.view(-1)

            intersection=(pred*mask).sum()
            dice=1-(2.0*intersection)/(pred.sum()+mask.sum())
            bce=nn.BCELoss()
            BCE=bce(pred,mask)
            dice_bce=BCE+dice
            return dice_bce
            



            
        


























