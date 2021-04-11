from torch import nn
from utils.anchor import anchor_generator 

class RegionProposalNetwork(nn.Module):
    def __init__(self,in_channels=512,mid_channels=512,ratios=[0.5,1,2],anchor_scales=[8,16,32],feat_stride=16,proposal_creator_params=dict()):
        super(RegionProposalNetwork,self).__init__()
        self.anchor_base = anchor_generator(ratios=ratios,anchor_scales=anchor_scales)
        self.feat_stride = feat_stride
        n_anchor=self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels,mid_channels,3,1,1)
        self.score = nn.Conv2d(mid_channels,n_anchor*2,1,1,0)
        self.loc = nn.Conv2d(mid_channels,n_anchor*4,1,1,0)
    

    def forward(self,x,)