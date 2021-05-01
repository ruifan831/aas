from torch import nn
from torchvision.ops import RoIPool
import torch
import numpy as np
class RoIHead(nn.Module):
    def __init__(self,n_class,classifier,roi_size,spatial_scale):
        super(RoIHead,self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096,n_class*4)
        self.score = nn.Linear(4096,n_class)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size,self.roi_size),self.spatial_scale)
    
    def forward(self,x,rois,roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices.view(-1,1), rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x,indices_and_rois)
        print("pool shape",pool.shape)
        pool = pool.view(pool.size(0),-1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs,roi_scores

def totensor(data, cuda=False):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor