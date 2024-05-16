
import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import json
import os
from data.split import split_test
import sys
from PIL import Image
from torchvision.transforms import Compose,ToTensor
import numpy as np
from torchvision.transforms import Resize,Compose,Pad,ToPILImage
from torchvision import transforms
import numpy as np
np.random.seed(0)
def bulid_dataset(path,config):
    transform=Compose([Resize([config["height"],config["width"]]),ToTensor()])
    transform_target=Compose([Resize([config["height"],config["width"]]),ToTensor()])
    train_dataset=CloudDataset(frames=config['frames'],interval=config["interval"],path=path,mode='train',transform=transform,transform_target=transform_target,series=config["series"])
    valid_dataset=CloudDataset(frames=config['frames'],interval=config["interval"],path=path,mode='valid',transform=transform,transform_target=transform_target,series=config["series"])
    return train_dataset,valid_dataset

class CloudDataset(Dataset):
    '''
    the dataset for the strong convective cloud
    input:
        frames      Timesteps of the sequence,which means the size of the sequence
        interval    The time interval of two picture in the sequence
        path        The path of images and labels
        mode        Dataset mode which decides train ,valid or test
        region      0:China  1:South 2:North
        patch       overlapping patch or resize
    '''
    def __init__(self,frames=6,interval=15,path='./data/MCSRSI',mode='train',transform=None,transform_target=None,
                date:str=None,series=True) -> None:
        super().__init__()
        self.series=series
        if date:
            self.name=[date]
        else:
            self.name=[str(i) for i in range(201801,201813)]
        self.image=path+'/bright_images'
        self.labels=path+'/labels_v2/all'
        if transform:
            self.transform=transform
        else:
            self.transform=Compose([ToTensor()])
        if transform_target:
            self.transform_target=transform_target
        else:
            self.transform_target=Compose([ToTensor()])
        
        # split the origin data into train,valid,test by month
        split_test(frames=frames,timesteps=interval,path='./tmp',series=series)
        with open('./tmp.json','r') as f:
            data=json.load(f)
        # if not os.path.exists(f'{path}/frame{frames}_interval{interval}.json'):
        #     raise Exception(f'{path}/frame{frames}_interval{interval}.json not exists, please manually divide the dataset by using split_test in MCSRSI/split.py')
        # with open(f'{path}/frame{frames}_interval{interval}.json','r') as f:
        #     data=json.load(f)
        self.data=[]
        for key in self.name:
            for t in data[key][mode]:
                self.data.append(t)
    def __getitem__(self, index):
        path=self.data[index]
        name=path[0][0:6]
        # generate the sequence and label
        image=[]
        label=[]
        # China 
        for i in range(len(path)):
            # Image
            image_path=self.image+"/"+name+"/"+path[i].split('.')[0]+'_bright_img.png'
            data=Image.open(image_path)
            image.append(self.transform(data))
            # Label
            label_path=self.labels+"/"+name+"/"+path[i]
            tmp_label=Image.open(label_path)
            tmp_label=self.transform_target(tmp_label)
            if torch.max(tmp_label)>torch.tensor(0.0):
                 tmp_label=(tmp_label-torch.min(tmp_label))/torch.max(tmp_label)
            label.append(tmp_label)
        image=torch.stack(image)
        label=torch.stack(label)
        if image.shape[0]==1:
            image=torch.unbind(image)[0]
            label=torch.unbind(label)[0]
        elif not self.series:
            label=torch.unbind(label)[-1]
        return image,label
    
    def __len__(self):
        return len(self.data)
    def getName(self,i):
        return self.data[i]

if __name__=='__main__':
    pass

        