import numpy as np
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
    w = base_size*np.sqrt(1/np.array(ratio))
    return h,w



