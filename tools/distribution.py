import os
import sys
import torch
from PIL import Image
from torchvision.transforms import Resize,ToTensor,Compose
if __name__=='__main__':
    path='/data/Jiatoka/example/spring/20180311124500_20180311124916.png'
    img=Image.open(path)
    model=Compose([Resize((160,256)),ToTensor()])
    img=model(img)
    print(img.shape)
    img=(img-torch.min(img))/(torch.max(img)-torch.min(img))
    img[img>=0.5]=1
    img[img<=0.5]=0
    data=torch.where(img==1)
    cnt=len(data[0])
    total=img.shape[-1]*img.shape[-2]
    print(cnt/total)