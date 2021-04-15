from data.dataset import VOCDataset
from torch.utils.data import DataLoader

root = root = "/home/ruifanxu/Desktop/ComputerVision/Faster_RCNN/VOCdevkit/VOC2007/"
trainData = VOCDataset(root)
loader = DataLoader(trainData,batch_size = 1)