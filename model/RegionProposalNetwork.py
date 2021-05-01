from torch import nn
import torch.nn.functional as F
from .utils.anchor import AnchorGenerator
from .utils.creator_tool import ProposalCreator
import torch


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(
            in_channels, num_anchors*2, kernel_size=1, stride=1)
        self.offset_pred = nn.Conv2d(
            in_channels, num_anchors*4, kernel_size=1, stride=1)

    def forward(self, x):
        h = F.relu(self.conv(x))
        cls_pred = self.cls_pred(h)
        offset_pred = self.offset_pred(h)
        return cls_pred, offset_pred


class RegionProposalNetwork(nn.Module):
    def __init__(self, rpn_anchor_generator, head,proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = rpn_anchor_generator
        self.rpn_head = head
        self.proposal_layer = ProposalCreator(self,**proposal_creator_params)

    def forward(self, x, img_shape, scale):
        n,_,h,w=x.shape
        anchors = self.anchor_generator(img_shape, x)
        print("anchors",anchors.shape)
        n_anchor = anchors.shape[1]//(h*w)
        rpn_score,rpn_offset = self.rpn_head(x)
        rpn_offset=rpn_offset.view(n,-1,4)
        rpn_score = rpn_score.permute(0,2,3,1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_score.view(n,h,w,n_anchor,2),dim=4)
        rpn_fg_scores=rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_fg_scores=rpn_fg_scores.view(n,-1)
        rpn_score = rpn_score.view(n,-1,2)

        rois=[]
        roi_indices=list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_offset[i],
                rpn_fg_scores[i],
                anchors[0],img_shape[i].shape[-2:],scales=scale
            )
            rois.append(roi)
            batch_index= i * torch.ones((len(roi),),dtype=torch.int32)
            roi_indices.append(batch_index)
        rois = torch.cat(rois,dim=0)
        print(rois.shape)
        roi_indices = torch.cat(roi_indices,dim=0)
        return rpn_offset,rpn_score,rois,roi_indices, anchors
        

