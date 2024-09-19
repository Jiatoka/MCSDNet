import os
import sys
import json
if __name__=='__main__':
    path="/data/Jiatoka/MCSDNet/tools/valid.json"
    with open(path,"r") as f:
        data=json.load(f)
    with open(path,"w") as f:
        json.dump(data,f,indent=1)