import numpy as np
import torch as t
from torch import nn

class AnchorGenerator(nn.Module):
    def __init__(self,ratios=[[0.5,1,2]],anchor_scales=[[8,16,32]],base_size=16):
        super().__init__()
        self.ratios = ratios
        self.anchor_scales= anchor_scales
        self.base_size= base_size
        self.cell_anchors =None
    
    def _ratio_enum(self,ratios):
        h = self.base_size*t.sqrt(ratios)
        w = self.base_size*t.sqrt(1/ratios)
        return h,w
    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.anchor_scales)]
    
    def generate_anchors(self,ratios,scales,dtype=t.float32,device="cpu"):
        ratios = t.as_tensor(ratios, dtype=dtype, device=device)
        anchor_scales = t.as_tensor(scales, dtype=dtype, device=device)
        h,w = self._ratio_enum(ratios)
        hs = (anchor_scales.view(-1,1)*h.view(1,-1)).view(-1)
        ws = (anchor_scales.view(-1,1)*w.view(1,-1)).view(-1)
        base_anchor = t.stack((-hs,-ws,hs,ws),dim=1)/2
        return base_anchor
    
    def set_cell_anchors(self,dtype,device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return
        cell_anchors=[self.generate_anchors(ratios,scales,dtype,device) for ratios,scales in zip(self.ratios,self,anchor_scales)]
        self.cell_anchors = cell_anchors
    def forward(self,imgs,feature_maps):
        # feature_map: (N,C,H,W)
        grid_sizes=[feature_map.shape[-2:] for feature_map in feature_maps] # (N,2)
        img_size = imgs.tensors.shape[-2:] # (N，2）
        dtype,device = feature_maps[0].dtype,feature_maps[0].device
        strides= [[t.tensor(img_size[0]//g[0],dtype=t.int64,device=device),
                  t.tensor(img_size[1]//g[1],dtype=t.int64,device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype,device)
        anchors_over_feature_maps = self.grid_anchors(grid_sizes,strides)
        return anchors_over_feature_maps
    
    def grid_anchors(self,grid_sizes,strides):
        anchors = []
        for size,stride in zip(grid_sizes,strides):
            grid_height,grid_width = size
            stride_height,stride_width = stride
            device = self.cell_anchors.device
            shifts_y = t.arange(0,grid_height,dtype=t.float32,device=device)*stride_height
            shifts_x = t.arange(0,grid_width,dtype=t.float32,device=device)*stride_width
            shift_x,shift_y = t.meshgrid(shifts_x,shifts_y)
            shift_y = shift_y.reshape(-1)
            shift_x = shift_x.reshape(-1)
            shifts = t.stack((shift_y,shift_x,shift_y,shift_y),dim=1)
            anchors.append((shifts.view(-1,1,4)+self.cell_anchors.view(1,-1,4)).reshape(-1,4))
        # anchors: (N,number_of_grid *9,4)
        return t.as_tensor(anchors)


