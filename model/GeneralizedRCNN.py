from torch import nn
from .utils.anchor import AnchorGenerator


class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        original_image_sizes = []
        featureMap = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, featureMap, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)


class FasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=None, min_size=800, max_size=1333, image_mean=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
                 out_channels = backbone.out_channels

                 if rpn_anchor_generator is None:
                     anchor_sizes = ((32,),(64,),(128,),(512))
                     aspect_ratios = ((0.5,1.0,2.0),)*len(anchor_sizes)
                     
