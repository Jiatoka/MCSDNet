import os
import sys
sys.path.append(os.getcwd())
from net.module.attention.attention2d import *
from net.module.attention.attention3d import *
from net.module.attention.positionEncoding import *
def create_attention(attn_name,*args,**kwargs):
    if attn_name=='msa':
        return MSA(dim=kwargs['dim'],head=kwargs['head'],dropout=kwargs['dropout'])
    elif attn_name=='rope_msa':
        return RotaryPEMultiHeadAttention(heads=kwargs['head'],d_model=kwargs['dim'],
        dropout_prob=kwargs['dropout'])
    else:
        raise Exception(f"The attention {attn_name} doesn't exist")
if __name__=='__main__':
    attn=create_attention(attn_name='rope_msa',dim=768,head=8,dropout=0.0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn=attn.to(device)
    data=torch.randn(4,6,768).to(device)
    out=attn(data)
    print(out.shape)
