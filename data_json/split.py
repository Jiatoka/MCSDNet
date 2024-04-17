# 划分脚本按照不同的frame和时间间隔划分数据
# 测试集，验证集，训练集的划分都是按月份来，保证0.15：0.15：0.7
import os
import sys
sys.path.append(os.getcwd())
import argparse
import json
import datetime
from sklearn.model_selection import KFold,train_test_split
import numpy as np
np.random.seed(0)
def split_test(frames,timesteps,path='./test',series=False):
    '''
    非严格等时间间隔
    time:0 全年 1:上半年  2:下半年
    '''
    with open("./data_json/labels.json") as f:
        data=json.load(f)
    train_output={}
    test_output={}
    output={}
    length=0
    if series==False:
        for key in data:
            output[key]=[]
            tmp=data[key]
            # 滑动窗口划分序列
            for l in range(len(tmp)):
                t=[]
                for r in range(l,len(tmp),int(timesteps/15)):
                    t.append(tmp[r])
                    if len(t)==frames:
                        break
                if len(t)==frames:
                    length+=1
                    output[key].append(t)
            real_output={}
        for key in output:
            real_output[key]={}
            tmp=np.array(output[key])
            index=np.arange(start=0,stop=len(tmp))
            train_valid,test=train_test_split(index,test_size=0.15,random_state=0)
            train,val=train_test_split(train_valid,test_size=0.15/0.85,random_state=0)
            real_output[key]['train'],real_output[key]['valid'],real_output[key]['test']=tmp[train].tolist(),tmp[val].tolist(),tmp[test].tolist()
        with open(f"{path}.json",'w') as f:
            json.dump(real_output,f,indent=1)
    else:
        # 每个月划分为5份,取其中一份做测试集合 
        for key in data:
            test_output[key]=[]
            train_output[key]=[]
            tmp=data[key]
            batch_len=len(tmp)//5 
            test_index=np.random.randint(0,5)
            test_tmp=tmp[test_index*batch_len:(test_index+1)*batch_len]
            train_tmp1=tmp[0:test_index*batch_len]
            train_tmp2=tmp[(test_index+1)*batch_len:]
            # 滑动窗口划分序列
            for l in range(len(test_tmp)):
                t=[]
                for r in range(l,len(test_tmp),int(timesteps/15)):
                    t.append(test_tmp[r])
                    if len(t)==frames:
                        break
                if len(t)==frames:
                    length+=1
                    test_output[key].append(t)
            for l in range(len(train_tmp1)):
                t=[]
                for r in range(l,len(train_tmp1),int(timesteps/15)):
                    t.append(train_tmp1[r])
                    if len(t)==frames:
                        break
                if len(t)==frames:
                    length+=1
                    train_output[key].append(t)
            
            for l in range(len(train_tmp2)):
                t=[]
                for r in range(l,len(train_tmp2),int(timesteps/15)):
                    t.append(train_tmp2[r])
                    if len(t)==frames:
                        break
                if len(t)==frames:
                    length+=1
                    train_output[key].append(t)
        real_output={}
        for key in train_output:
            real_output[key]={}
            real_output[key]['train'],real_output[key]['valid'],real_output[key]['test']=train_output[key],test_output[key],test_output[key]
        with open(f"{path}.json",'w') as f:
            json.dump(real_output,f,indent=1)
    return length
if __name__=='__main__':
    '''
    "20180118024500_20180118025959.png",
    "20180118031500_20180118032959.png",
    "20180118034500_20180118034916.png",
    "20180118041500_20180118041916.png",
    "20180118044500_20180118044916.png",
    "20180118051500_20180118051916.png"
    '''
    parse=argparse.ArgumentParser(description="divide the dataset")
    parse.add_argument('-p','--path',type=str,default='./tmp',help='json output path')
    parse.add_argument('-t','--timesteps',type=int,default=15,help='intervals between images')
    parse.add_argument('-f','--frames',type=int,default=1,help='frames in sequence')
    args=parse.parse_args()
    frames=args.frames
    timesteps=args.timesteps
    path=args.path
    timesteps=120
    frames=6
    length=split_test(frames=frames,timesteps=timesteps,path=path,series=True)

                    


