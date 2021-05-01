import torch

def smooth_l1_loss(rpn_offset,offset,label):
    in_weight = torch.zeros(offset.shape)
    in_weight[(label>0).view(-1,1).expand_as(in_weight)]=1
    differences = (offset-rpn_offset)*in_weight
    abs_differences = differences.abs()
    flag = (abs_differences<1).float()
    y = flag*(abs_differences**2)+(1-flag)*(abs_differences-0.5)
    return y.sum()