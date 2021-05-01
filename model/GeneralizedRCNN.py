from torch import nn
from .utils.anchor import AnchorGenerator
from .ROIHead import RoIHead
from .RegionProposalNetwork import RPNHead,RegionProposalNetwork
from .Backbone import vggbackbone

class FasterRCNN(nn.Module):
    def __init__(self, backbone,rpn,head):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.head = head
    
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
        #     self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

class FasterRCNN_vgg16(FasterRCNN):

    feat_stride = 16

    def __init__(self, num_classes=None, min_size=800, max_size=1333, rpn_anchor_generator=None, rpn_head=None, rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100, box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512, box_positive_fraction=0.25, bbox_reg_weights=None):
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
                
                
                
                     
