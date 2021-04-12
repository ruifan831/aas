import numpy as np
import torch as t

class AnchorGenerator(nn.Module):
    def __init__(self,ratios=[0.5,1,2],anchor_scales=[8,16,32],base_size=16):
        super().__init__()
        self.ratios = ratios
        self.anchor_scales= anchor_scales
        self.base_size= base_size
    
    def _ratio_enum(self,ratios):
        h = self.base_size*t.sqrt(self.ratios)
        w = self.base_size*t.sqrt(1/t.as_tensor(self.ratios))
        return h,w
    
    def generate_anchors(self,dtype=torch.float32,device="cpu"):
        h,w = self._ratio_enum()




def anchor_generator(ratios=[0.5,1,2],anchor_scales=[8,16,32],base_size=16):
    h,w = _ratio_enum(base_size, ratios)
    x_ctr = (base_size-1)/2
    y_ctr = (base_size-1)/2
    anchor_base = np.zeros((len(ratios)*len(anchor_scales),4),dtype = np.float32)
    for i,scale in enumerate(anchor_scales):
        for j in range(3):
            index = i*len(anchor_scales) +j
            anchor_base[index,0] = y_ctr - (h[j]*scale-1)/2
            anchor_base[index,1] = x_ctr - (w[j]*scale-1)/2
            anchor_base[index,2] = y_ctr + (h[j]*scale-1)/2
            anchor_base[index,3] = x_ctr + (w[j]*scale-1)/2
    return anchor_base
             



def _ratio_enum(base_size,ratios):
    h = base_size*np.sqrt(ratios)
    w = base_size*np.sqrt(1/np.array(ratios))
    return h,w



