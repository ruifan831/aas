import torch.nn as nn
import torch.nn.functional as F

from model.utils.creator_tool import ProposalCreator,ProposalTargetCreator,AnchorTargetGenerator
import torch

import time

def smooth_l1_loss(rpn_offset,offset,label):
    in_weight = torch.zeros(offset.shape)
    in_weight[(label>0).view(-1,1).expand_as(in_weight)]=1
    differences = (offset-rpn_offset)*in_weight
    abs_differences = differences.abs()
    flag = (abs_differences<1).float()
    y = flag*(abs_differences**2)+(1-flag)*(abs_differences-0.5)
    return y.sum()/(label>=0).sum().float()

class FasterRCNNTrainer(nn.Module):
    def __init__(self,faster_rcnn):
        super(FasterRCNNTrainer,self).__init__()
        self.faster_rcnn = faster_rcnn
        self.anchor_target_creator = AnchorTargetGenerator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.optimizer = self.faster_rcnn.get_optimizer()

    
    def forward(self,imgs,target):

        features = self.faster_rcnn.backbone(imgs)

        rpn_offsets, rpn_scores, rois,roi_indices,anchors=self.faster_rcnn.rpn(features,imgs,1)

        bbox = target["boxes"][0]
        label = target["labels"][0]
        rpn_score = rpn_scores[0]
        rpn_offset = rpn_offsets[0]

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(rois, bbox, label)
        sample_roi_index= torch.zeros(len(sample_roi))
        roi_cls_loc,roi_score = self.faster_rcnn.head(features,sample_roi,sample_roi_index)

        # ---- ROI LOSS ---- #
        gt_roi_loc = torch.from_numpy(gt_roi_loc)
        roi_cls_loss = F.cross_entropy(roi_score,gt_roi_label)
        n_sample = gt_roi_loc.shape[0]

        roi_cls_loc = roi_cls_loc.view(n_sample,-1,4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(),gt_roi_label]
        roi_offset_loss = smooth_l1_loss(roi_loc,gt_roi_loc,gt_roi_label)

        # ---- RPN LOSS ---- #
        offset,label = self.anchor_target_creator(bbox,anchors[0],imgs[0].shape[-2:])
        label = torch.from_numpy(label).long()
        offset = torch.from_numpy(offset)
        rpn_cls_loss = F.cross_entropy(rpn_score.view(-1,2),label,ignore_index=-1)
        rpn_offset_loss = smooth_l1_loss(rpn_offset.view(-1,4),offset,label)

        losses = [rpn_offset_loss, rpn_cls_loss, roi_offset_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return losses
    
    def train_step(self,imgs,target):
        self.optimizer.zero_grad()
        losses = self.forward(imgs,target)
        losses[-1].backward()
        self.optimizer.step()
        return losses
    

    def save(self,save_path=None):
        if save_path is None:
            timestr = time.strftime("%Y_%m_%d_%H_%M")
            save_path = f"checkpoints/faster_rcnn_{timestr}"
        
        save_dir = os.path.dirname(save_path)
        if not os.exists(save_dir):
            os.makedirs(save_dir)
        save_dict=dict()
        save_dict["model"] = self.faster_rcnn.state_dict()
        torch.save(save_dict,save_path)
        return save_path
    
    def load(self,path):
        checkpoint = torch.load(path)
        if "model" in checkpoint:
            self.faster_rcnn.load_state_dict(checkpoint["model"])
        return self


