from torch import nn
import torch.nn.functional as F
from utils.anchor import anchor_generator



class RPNHead(nn.Module):
    def __init__(self,in_channels,num_anchors):
        super(RPNHead,self).__init__()
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size = 3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(in_channels,num_anchors*2,kernel_size = 1,stride =1)
        self.offset_pred = nn.Conv2d(in_channels, num_anchors*4, kernel_size=1, stride=1)
    
    def forward(self,x):
        h = F.relu(self.conv(x))
        cls_pred = self.cls_pred(h)
        offset_pred = self.offset_pred(h)
        return cls_pred,offset_pred

class RegionProposalNetwork(nn.Module):
    def __init__(self,in_channels=512,mid_channels=512,ratios=[0.5,1,2],anchor_scales=[8,16,32],feat_stride=16,proposal_creator_params=dict()):
        super(RegionProposalNetwork,self).__init__()
        self.anchor_base = anchor_generator(ratios=ratios,anchor_scales=anchor_scales)
        self.feat_stride = feat_stride
        n_anchor=self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels,mid_channels,3,1,1)
        self.score = nn.Conv2d(mid_channels,n_anchor*2,1,1,0)
        self.loc = nn.Conv2d(mid_channels,n_anchor*4,1,1,0)
    

    def forward(self,x,img_size,scale=1):
        n,_,h,w = x.shape
        anchors = generate_anchors(self.anchor_base,self.feat_stride,h,w)
        n_anchor = anchors.shape[0]//(h*w)
        h = F.relu(self.conv1(x))

        # rpn_score shape: (N, len(ratios)*len(anchor_scales)*4,H,W)
        rpn_offset = self.loc(h)
        rpn_offset = rpn_offset.permute(0,2,3,1).contiguous().view(n,-1,4)

        # rpn_score shape: (N, len(ratios)*len(anchor_scales)*2,H,W)
        rpn_score = self.score(h)
        rpn_score = rpn_score.permute(0,2,3,1).contiguous()
        rpn_softmax_score = F.softmax(rpn_score.view(n,h,w,n_anchor,2),dim=4)
        rpn_fg_scores = rpn_softmax_score[:,:,:,:,1].contiguous().view(n,-1)
        rpn_score = rpn_score.view(n,-1,2)

        rois = list()
        roi_indices = list()

        



def generate_anchors(anchor_base,feat_stride,height,width):
    import numpy as np
    shift_y = np.arange(0,height*feat_stride,feat_stride)
    shift_x = np.arange(0,width*feat_stride,feat_stride)
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    shift = np.stack((shift_y.ravel(),shift_x.ravel(),shift_y.ravel(),shift_x.ravel()),axis=1)
    A = anchor_base.shape[0]
    # K is the number of coordinate that will have anchor on it.
    K = shift.shape

    anchor = anchor_base.reshape((1,A,4))+shift.reshape((K,1,4))
    anchor = anchor.reshape((K*A,4)).astype(np.float32)
    return anchor

