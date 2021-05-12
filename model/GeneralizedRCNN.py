from torch import nn
import torch
from torch.nn import functional as F

from .utils.anchor import AnchorGenerator
from .utils.creator_tool import offset2bbox
from .ROIHead import RoIHead
from .RegionProposalNetwork import RPNHead,RegionProposalNetwork
from .Backbone import vggbackbone
from torchvision.ops import nms
import numpy as np
from utils import arrayUtils


class FasterRCNN(nn.Module):
    def __init__(self, backbone,rpn,head):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.head = head
        self.use_preset("evaluate")
    def forward(self,x,scale=1):
        h = self.backbone(x)
        rpn_offset, rpn_score, rois,roi_indices,anchors = self.rpn(h,x,scale)
        roi_cls_loc,roi_score = self.head(h,rois,roi_indices)
        return roi_cls_loc,roi_score,rois,roi_indices
    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = 1e-3
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        # if opt.use_adam:
        #     self.optimizer = t.optim.Adam(params)
        # else:
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer
    @property
    def n_class(self):
        return self.head.n_class

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')
    def suppress(self,raw_cls_bbox,raw_prob):
        bbox = []
        label = []
        score = []
        for i in range(1,self.n_class):
            offset_i = raw_cls_bbox.reshape((-1,self.n_class,4))[:,i,:]
            prob_i = raw_prob[:,i]
            mask = prob_i>self.score_thresh
            offset_i = offset_i[mask] 
            prob_i = prob_i[mask]
            keep = nms(offset_i,prob_i,self.nms_thresh)
            bbox.append(offset_i[keep].cpu().numpy())
            label.append(i*np.ones((len(keep),)))
            score.append(prob_i[keep].cpu().numpy())
        print(len(bbox))
        
        bbox = np.concatenate(bbox,axis=0).astype(np.float32)
        label = np.concatenate(label,axis=0).astype(np.int32)
        score = np.concatenate(score,axis=0).astype(np.float32)
        return bbox,label,score
    
    def pred(self,imgs,sizes):
        self.eval()
        bboxes = []
        labels = []
        scores = []
        for img,size in zip(imgs,sizes):
            roi_cls_loc,roi_score,rois,_ = self(imgs)
            roi_score= roi_score.data
            roi_cls_loc = roi_cls_loc.data
            roi = rois
            roi_cls_loc = roi_cls_loc.view(-1,self.n_class,4)
            roi = roi.view(-1,1,4).expand_as(roi_cls_loc)
            result_bbox = offset2bbox(
                arrayUtils.tonumpy(roi).reshape((-1,4)),
                arrayUtils.tonumpy(roi_cls_loc).reshape((-1,4))
            )
            result_bbox = arrayUtils.totensor(result_bbox)
            result_bbox = result_bbox.view(-1,self.n_class*4)

            result_bbox[:,0::2] = result_bbox[:,0::2].clamp(min=0,max=size[0])
            result_bbox[:,1::2] = result_bbox[:,1::2].clamp(min=0,max=size[1])

            prob = F.softmax(roi_score,dim=1)
            print(result_bbox.shape)

            bbox,label,score = self.suppress(result_bbox,prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        
        self.use_preset("evaluate")
        self.train()
        return bboxes,labels,scores

class FasterRCNN_vgg16(FasterRCNN):

    feat_stride = 16

    def __init__(self, num_classes=None, min_size=800, max_size=1333, rpn_anchor_generator=None, rpn_head=None, rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300, rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None):
            backbone, classifier = vggbackbone()
            out_channels = backbone.backbone[-2].out_channels
            if rpn_anchor_generator is None:
                anchor_sizes = [[8,16,32]]
                aspect_ratios = [[0.5,1.0,2.0],]
                rpn_anchor_generator = AnchorGenerator(aspect_ratios,anchor_sizes)

            if rpn_head is None:
                rpn_head = RPNHead(out_channels,rpn_anchor_generator.num_anchors_per_location()[0])
                
            proposal_creator_params = {
                "nms_thresh":rpn_nms_thresh,
                "n_train_pre_nms":rpn_pre_nms_top_n_train,
                "n_train_post_nms":rpn_post_nms_top_n_train,
                "n_test_pre_nms":rpn_pre_nms_top_n_test,
                "n_test_post_nms":rpn_post_nms_top_n_test,
                "min_size":16
            }
            rpn = RegionProposalNetwork(rpn_anchor_generator,rpn_head,proposal_creator_params)

            roi_head = RoIHead(num_classes,classifier,7,1/self.feat_stride)
            super(FasterRCNN_vgg16,self).__init__(backbone,rpn,roi_head)
                
                
                
                     
