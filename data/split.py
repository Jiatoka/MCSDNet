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
    with open("./data/labels.json") as f:
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
def splitTest(frames,timesteps,series=False):
    '''
    非严格等时间间隔
    '''
    # with open(f"../data/labelsTest.json") as f:
    #         data=json.load(f)
    data={
 "test": [
  "20180710231500_20180710231916.png",
  "20180711013000_20180711013416.png",
  "20180711023000_20180711023416.png",
  "20180711031500_20180711032959.png",
  "20180711033000_20180711033416.png",
  "20180711110000_20180711111459.png",
  "20180711111500_20180711111916.png",
  "20180711113000_20180711113416.png",
  "20180711120000_20180711121459.png",
  "20180711124500_20180711124916.png",
  "20180711143000_20180711143416.png",
  "20180711150000_20180711151459.png",
  "20180711180000_20180711181459.png",
  "20180711181500_20180711182959.png",
  "20180711203000_20180711203416.png",
  "20180712010000_20180712011459.png",
  "20180712054500_20180712055959.png",
  "20180712070000_20180712071459.png",
  "20180712113000_20180712113416.png",
  "20180712144500_20180712145959.png",
  "20180712151500_20180712152959.png",
  "20180712154500_20180712154916.png",
  "20180712163000_20180712163416.png",
  "20180712173000_20180712173416.png",
  "20180712200000_20180712201459.png",
  "20180712230000_20180712231459.png",
  "20180712234500_20180712235959.png",
  "20180713003000_20180713003416.png",
  "20180713011500_20180713011916.png",
  "20180713033000_20180713033416.png",
  "20180713034500_20180713034916.png",
  "20180713110000_20180713111459.png",
  "20180713120000_20180713121459.png",
  "20180713121500_20180713122959.png",
  "20180713130000_20180713131459.png",
  "20180713133000_20180713133416.png",
  "20180713134500_20180713134916.png",
  "20180713140000_20180713141459.png",
  "20180713143000_20180713143416.png",
  "20180713150000_20180713151459.png",
  "20180713151500_20180713152959.png",
  "20180713174500_20180713175959.png",
  "20180713193000_20180713193416.png",
  "20180713201500_20180713201916.png",
  "20180713203000_20180713203416.png",
  "20180713220000_20180713221459.png",
  "20180713234500_20180713235959.png",
  "20180714034500_20180714034916.png",
  "20180714041500_20180714041916.png",
  "20180714050000_20180714051459.png",
  "20180714113000_20180714113416.png",
  "20180714131500_20180714131916.png",
  "20180714144500_20180714145959.png",
  "20180714150000_20180714151459.png",
  "20180714180000_20180714181459.png",
  "20180714194500_20180714194916.png",
  "20180714230000_20180714231459.png",
  "20180715011500_20180715011916.png",
  "20180715014500_20180715014916.png",
  "20180715024500_20180715025959.png",
  "20180715031500_20180715032959.png",
  "20180715040000_20180715041459.png",
  "20180715051500_20180715051916.png",
  "20180715063000_20180715063416.png",
  "20180715104500_20180715104916.png",
  "20180715121500_20180715122959.png",
  "20180715141500_20180715141916.png",
  "20180715150000_20180715151459.png",
  "20180715174500_20180715175959.png",
  "20180715234500_20180715235959.png",
  "20180716014500_20180716014916.png",
  "20180716023000_20180716023416.png",
  "20180716063000_20180716063416.png",
  "20180716073000_20180716073416.png",
  "20180716083000_20180716083416.png",
  "20180716090000_20180716091459.png",
  "20180716091500_20180716092959.png",
  "20180716100000_20180716101459.png",
  "20180716110000_20180716111459.png",
  "20180718003000_20180718003416.png",
  "20180718004500_20180718004916.png",
  "20180718011500_20180718011916.png",
  "20180718013000_20180718013416.png",
  "20180718020000_20180718021459.png",
  "20180718030000_20180718031459.png",
  "20180718034500_20180718034916.png",
  "20180718093000_20180718093416.png",
  "20180718130000_20180718131459.png",
  "20180718133000_20180718133416.png",
  "20180718164500_20180718164916.png",
  "20180718170000_20180718171459.png",
  "20180718194500_20180718194916.png",
  "20180718200000_20180718201459.png",
  "20180719011500_20180719011916.png",
  "20180719054500_20180719055959.png",
  "20180719060000_20180719061459.png",
  "20180719063000_20180719063416.png",
  "20180719064500_20180719064916.png"
 ]
}
    output={}
    length=0
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
    with open('./tmpTest.json','w') as f:
        json.dump(output,f,indent=1)
if __name__=='__main__':
    '''
    "20180118024500_20180118025959.png",
    "20180118031500_20180118032959.png",
    "20180118034500_20180118034916.png",
    "20180118041500_20180118041916.png",
    "20180118044500_20180118044916.png",
    "20180118051500_20180118051916.png"
    '''
    parse=argparse.ArgumentParser(description="划分数据集合")
    parse.add_argument('-p','--path',type=str,default='./tmp',help='json输出路径')
    parse.add_argument('-t','--timesteps',type=int,default=15,help='图片帧之间的时间间隔')
    parse.add_argument('-f','--frames',type=int,default=1,help='一个序列中的帧数')
    args=parse.parse_args()
    frames=args.frames
    timesteps=args.timesteps
    path=args.path
    timesteps=30
    frames=6
    length=split_test(frames=frames,timesteps=timesteps,path=path,series=True)
    print("序列长度:",length)
    with open(f"{path}.json",'r') as f:
        data=json.load(f)
        train_len=0
        valid_len=0
        train_series=0
        valid_series=0
        for time in data:
            train=[]
            valid=[]
            test=[]
            for j in data[time]['train']: 
                train.extend(j)
            for j in data[time]['valid']:
                valid.extend(j)
            train_series+=len(data[time]['train'])
            valid_series+=len(data[time]['valid'])
            train_len+=len(set(train))
            valid_len+=len(set(valid))
            print(f"{time}_frames{frames}_interval{timesteps}_train:{len(set(train))}")
            print(f"{time}_frames{frames}_interval{timesteps}_valid:{len(set(valid))}")
        print(f"train_len:{train_len}")
        print(f"valid_len:{valid_len}")
        print(f"train_series:{train_series}")
        print(f"valid_series:{valid_series}")


                    


