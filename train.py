from data.dataset import VOCDataset,Transform
from torch.utils.data import DataLoader
import torch
from trainer import FasterRCNNTrainer
from model.GeneralizedRCNN import FasterRCNN_vgg16
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
root="../VOCdevkit/VOC2007/"


def train():
    dataset = VOCDataset(root,transform = Transform())
    print("Loading Data")
    dataLoader = DataLoader(dataset,batch_size=1,shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rcnn = FasterRCNN_vgg16(num_classes=21)
    print("Model Constructed")
    trainer = FasterRCNNTrainer(rcnn).to(device)
    for epoch in range(10):
        for i,(img,target) in tqdm(enumerate(dataLoader)):
            img = img.to(device)
            losses = trainer.train_step(img,target)
            print(f"RPN OFFSET LOSSES: {losses[0]}  |RPN CLASSIFICATION LOSS: {losses[1]}\nROI OFFSET LOSSES: {losses[2]}  |ROI CLASSIFICATION LOSS: {losses[3]}")
                


if __name__ == '__main__':
    train()


