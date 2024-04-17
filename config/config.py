import torch
from typing import Union
import yaml
def load_config(path:Union[str,None]):
    if path is None:
        return None
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data
if __name__=='__main__':
    config=load_config("/workplace/project/Convective/config/tsunet.yaml")
    print(config['model']['v1']['dataset'])
