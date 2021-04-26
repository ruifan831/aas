from .anchor import anchorWithOffset
import torch
from torchvision.ops import nms
import numpy as np

def bbox_iou(roi,bbox):
    """
    Args:
        roi (numpy array):  the region of interests with shape of (M,4)
        bbox (numpy array): the ground true bounding boxes with shape of (Number of bounding boxes in the image,4)
    
    Returns:
        A numpy array with shape of (M,Number of bounding boxes in the image), each value is the iou percentage between each roi and each bounding boxes.
    """
    top_left_intersection = np.maximum(roi[:,None,:2],bbox[:,:2])
    bottom_right_intersection = np.minimum(roi[:,None,2:],bbox[:,2:])
    
    area_i = np.prod(bottom_right_intersection-top_left_intersection,axis=2)*(top_left_intersection<bottom_right_intersection).all(axis=2)
    area_roi = np.prod(roi[:,2:] - roi[:,:2],axis=1)
    area_bbox = np.prod(bbox[:,2:] - bbox[:,:2],axis=1)
    return area_i / (area_roi.reshape(-1,1)+area_bbox.reshape(1,-1)-area_i)


def bbox2offset(roi,bbox):
    height = roi[:,2] - roi[:,0]
    width = roi[:,3] - roi[:,1]
    ctr_y = roi[:,0] + 0.5*height
    ctr_x = roi[:,1] + 0.5*width

    bbox_height =bbox[:,2] - bbox[:,0]
    bbox_width = bbox[:,3] - bbox[:,1]
    bbox_ctr_y = bbox[:,0] + 0.5*height
    bbox_ctr_x = bbox[:,1] + 0.5*width

    dy = (bbox_ctr_y - ctr_y) / height
    dx = (bbox_ctr_x - ctr_x) / width
    dh = np.log(bbox_height / height)
    dw = np.log(bbox_width / width)
    
    loc = np.stack((dy,dx,dh,dw),axis=1)
    return loc


def tonumpy(data):
    if isinstance(data,np.ndarray):
        return data
    if isinstance(data,torch.Tensor):
        return data.detach().cpu().numpy()

def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class ProposalTargetCreator():
    def __init__(self,n_sample=128, pos_ratio = 0.25, pos_iou_thresh = 0.5,neg_iou_thresh_hi=0.5,neg_iou_thresh_lo =0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
    
    def __call__(self,roi,bbox,label):
        n_bbox, _ = bbox.shape
        roi = torch.cat((roi,bbox),dim=0)
        

        pos_roi_per_image = round(self.n_sample*self.pos_iou_thresh)
        iou = bbox_iou(tonumpy(roi),tonumpy(bbox))
        
        # assign each roi to the bounding box with the highest iou
        ground_truth_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        print(max(ground_truth_assignment))
        gt_roi_label = label[ground_truth_assignment]

        pos_index = np.where(max_iou>self.pos_iou_thresh)[0]
        pos_roi_for_this_image = int(min(pos_roi_per_image,pos_index.size))

        if pos_index.size>0:
            pos_index = np.random.choice(pos_index,size=pos_roi_for_this_image,replace=False)
        
        neg_index= np.where((max_iou<self.neg_iou_thresh_hi)&(max_iou>self.neg_iou_thresh_lo))[0]
        neg_roi_for_this_image = self.n_sample-pos_roi_for_this_image
        neg_roi_for_this_image = int(min(neg_roi_for_this_image,neg_index.size))
        
        if neg_index.size>0:
            neg_index = np.random.choice(neg_index,size=neg_roi_for_this_image,replace=False)
        
        index = np.append(pos_index,neg_index)
        ground_truth_roi_label = gt_roi_label[index]
        ground_truth_roi_label[pos_roi_for_this_image:]=0


        sampled_roi = roi[index]

        gt_roi_offset = bbox2offset(tonumpy(sampled_roi),tonumpy(bbox[ground_truth_assignment[index]]))
        return sampled_roi,gt_roi_offset,gt_roi_label

def get_valid_index(anchor,h,w):
    valid_index = np.where( (anchor[:,0]>=0) & (anchor[:,1]>=0) & (anchor[:,2]<=h) & (anchor[:,3]<=w))[0]
    return valid_index

class AnchorTargetGenerator:
    def __init__(self,n_sample=256,pos_ratio = 0.5, pos_iou_thresh = 0.7,neg_iou_thresh=0.3):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_raio = pos_ratio
    
    def __call__(self,bbox,anchor,img_size):
        H,W = img_size
        n_anchor = len(anchor)
        valid_index = get_valid_index(anchor,H,W)
        print(valid_index)
        anchor = anchor[valid_index]
        print(anchor.shape)

        argmax_ious,label = self.create_label(valid_index,anchor,bbox)

        offset = bbox2offset(anchor,bbox[argmax_ious])
        label = _unmap(label, n_anchor, valid_index, fill=-1)
        offset = _unmap(offset, n_anchor, valid_index, fill=0)
        return offset,label


    def create_label(self, valid_index,anchor,bbox):
        label = np.empty((len(valid_index)),dtype=np.int32)
        label.fill(-1)

        argmax_ious,max_ious,gt_argmax_ious = self.calc_ious(anchor,bbox,valid_index)

        label[max_ious<self.neg_iou_thresh] = 0

        label[gt_argmax_ious] = 1

        label[max_ious>=self.pos_iou_thresh] = 1

        n_pos = int(self.pos_raio * self.n_sample)
        pos_index = np.where(label==1)[0]

        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index,size=(len(pos_index)-n_pos),replace=False)
            label[disable_index] = -1
        
        n_neg = self.n_sample - np.sum(label==1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index,size=(len(neg_index)-n_neg),replace=False)
            label[disable_index] = -1
        return argmax_ious, label


    
    def calc_ious(self,anchor,bbox,valid_index):
        # ious shape: (Number of anchor, number of bbox)
        ious = bbox_iou(tonumpy(anchor),tonumpy(bbox))
        # argmax_ious shape: (Number of anchor,) return the bbox index that has the largest iou ratio.
        argmax_ious = ious.argmax(axis=1)
        # max_ious shape: (Number of anchor,) the max iou ratio each anchor has.
        max_ious = ious[np.arange(len(valid_index)),argmax_ious]

        
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious,np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious,max_ious,gt_argmax_ious
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
        keep = torch.where((hs>=min_size)& (ws>=min_size))[0]
        roi = roi[keep,:]


        score=score[keep]

        order = torch.argsort(score,descending=True)
        if n_pre_nms>0:
            order = order[:n_pre_nms]
        roi= roi[order,:]

        score = score[order]
        keep = nms(roi,score,self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu()]
        return roi

        


        
    