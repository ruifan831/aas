from torch import nn
from torchvision.ops import RoIPool
import torch
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
        roi_indices = totensor(roi_indice).float()
        rois = totensor(rois).float()
        indices_and_rois = t.cat([roi_indices.view(-1,1), rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()
        pool = self.roi(x,indices_and_rois)

        return indices_and_rois,roi_indices,rois


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor