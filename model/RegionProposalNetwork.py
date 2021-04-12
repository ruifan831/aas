from torch import nn
import torch.nn.functional as F
from utils.anchor import AnchorGenerator


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
    def __init__(self, rpn_anchor_generator, head,fg_iou_thresh, bg_iou_thresh, batch_size_per_images, positive_fraction,pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = rpn_anchor_generator
        self.rpn_head = head

    def forward(self, x, img_shape):
        n,_,h,w=x.shape
        anchors = self.anchor_generator(img_shape, x)
        n_anchor = anchors.shape[1]
        rpn_score,rpn_offset = self.rpn_head(x)
        rpn_score = rpn_score.permute(0,2,3,1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_score.view(n,h,w,n_anchor,2),dim=4)
        rpn_fg_scores=rpn_softmax_scores[:,:,:,:,1].contiguous()
        rpn_score=rpn_fg_scores.view(n,-1)
        rpn_score = rpn_score.view(n,-1,2)
        

