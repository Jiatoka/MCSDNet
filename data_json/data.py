# 整合脚本，用于按月份生成初始json
from pathlib import Path
import os
import sys
import json
# 按照月份生成原始数据的json
def getName(root,name:list,out='./data.json',flag=False):
    data={}
    other={}
    other['test']=[]
    length=0
    for dir_name in name:
        path=os.path.join(root,dir_name)
        path=Path(path)
        data[str(dir_name)]=[]
        file_name=[]
        for file in os.listdir(path):
            file=str(file).split('/')[-1]
            index=file.split('_')[0]
            file_name.append([int(index),file])
        file_name.sort(key=lambda x:x[0])
        # length+=len(file_name)
        for text in file_name:
            if flag:
                if str(text[0])[-2:]!='00':
                    other['test'].append(text[1])
                    continue
            length+=1
            data[str(dir_name)].append(text[1])
    with open(out,'w') as f:
        json.dump(data,f,indent=1)
    return length
# 生成测试用的数据
def getTestName(root,out='./dataTest.json',flag=False):
    data={}
    other={}
    length=0
    path=Path(root)
    data['test']=[]
    file_name=[]
    for file in os.listdir(path):
        file=str(file).split('/')[-1]
        index=file.split('_')[0]
        file_name.append([int(index),file])
    file_name.sort(key=lambda x:x[0])
    for text in file_name:
        length+=1
        data['test'].append(text[1])
    with open(out,'w') as f:
        json.dump(data,f,indent=1)
    return length
# 判断labels是否都蕴含在images中
def isIn():
    labels='./data_json/labels.json'
    image='./data_json/data.json'
    with open(image,'r') as f:
        image=json.load(f)
    with open(labels,'r') as f:
        labels=json.load(f)
    data1=[]
    data2=[]
    for value in image.values():
        for i in value:
            data1.append(i.split('_')[0]+"_"+i.split('_')[1])
    for value in labels.values():
        for i in value:
            data2.append(i)
    for i in data2:
        if not (i in data2):
            print(i)
            return False    
    return True
if __name__=='__main__':
    '''
    该部分代码用于统计并过滤数据，将图片的名字以json的形式输出
    结论:
    1.bright_images的图片数量50000+，labels的图片数量为12307
    2.labels的图片全部都蕴含在bright_images中
    3.划分数据集使用labels进行划分，labels中的数据绝大多数都是间隔15min，少部分间隔4min，需要将间隔4min的去掉,一共有8000张图片
    '''
    # print("长度:",getTestName('/workplace/dataset/EasyDataset/labels','./data_json/labelsTest.json'))
    pass
