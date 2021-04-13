from anchor import anchorWithOffset
import torch
from torchvision.ops import nms
class ProposalCreator:

    def __init__(self,parent_model,nms_thresh=0.7,n_train_pre_nms=12000,n_train_post_nms=2000,n_test_pre_nms=6000,n_test_post_nms=300,min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    
    def __call__(self,offset,score,anchor,img_size,scales=1):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        # transform the anchor via offset output by RPNHead and clip bboxes to image
        roi = anchorWithOffset(anchor,offset)
        roi[:,slice(0,4,2)] = torch.clip(roi[:,slice(0,4,2)],0,img_size[0])
        roi[:,slice(1,4,2)] = torch.clip(roi[:,slice(1,4,2)],0,img_size[1])

        min_size = self.min_size*scales
        hs = roi[:,2]-roi[:,0]
        ws = roi[:,3] - roi[:,1]
        keep = torch.where(hs>=min_size&ws>=min_size)[0]
        roi = roi[keep,:]
        score=score[keep]

        order = torch.argsort(score,descending=True)
        if n_pre_nms>0:
            order = order[:n_pre_nms]
        roi= roi[order,:]
        score = score[order]
        keep = nms(roi.cuda,score,self.nms_thresh)
        roi = roi[keep.cpu()]
        return roi

        


        
    