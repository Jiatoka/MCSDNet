import sys
import os
sys.path.append(os.getcwd())
from fvcore.nn import FlopCountAnalysis
import torch
from net.model import build_model
from config.config import load_config
import torchprofile
import copy
from ptflops import get_model_complexity_info
def compute_train(model,data):
    optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-3)
    criterion=torch.nn.CrossEntropyLoss()
    target=copy.deepcopy(data)
    for i in range(10):
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 在训练过程中监控显存占用
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/(1024*1024*1024)} G")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/(1024*1024*1024)} G")


if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    config=load_config("./config/longmcsdnet15.yaml")
    config=config['model']['MCSDNet']
    model = build_model(config)
    input_tensor = tuple([12, 1, 1, 160, 256])  # t,b,c,h,w
    flops, params = get_model_complexity_info(model, input_tensor, as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    
    model=model.to(device)
    data=torch.randn(input_tensor)
    data=data.to(device)

    # 计算现存
    compute_train(model,data)