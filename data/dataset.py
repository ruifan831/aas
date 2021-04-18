from torch.utils.data import Dataset,DataLoader
import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
class VOCDataset(Dataset):
    
    def __init__(self,root_dir,transform = None,split="trainval"):
        self.root_dir = root_dir
        self.transform = transform
        data_path = os.path.join(root_dir,"ImageSets/Main/{0}.txt".format(split))
        with open(data_path,"r") as f:
            self.ids = [i.strip() for i in f.readlines()]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        idx = self.ids[idx]
        img_path = os.path.join(self.root_dir,"JPEGImages",idx+".jpg")
        img = transforms.Compose([transforms.ToTensor()])(Image.open(img_path).convert("RGB"))
        anno = ET.parse(os.path.join(self.root_dir,"Annotations",idx+".xml"))
        bbox = list()
        label = list()
        for obj in anno.findall("object"):
            bnd = obj.find("bndbox")
            bbox.append([int(bnd.find(tag).text)-1 for tag in ("ymin","xmin","ymax","xmax")])
            name = obj.find("name").text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name)+1)
        target = {}
        target["boxes"] = torch.as_tensor(bbox,dtype = torch.float32)
        target["labels"] = torch.as_tensor(label,dtype = torch.int64)
        target["image_id"] = idx
        if self.transform is not None:
            img,target = self.transform((img,target))
        return img, target

class Transform:
    def __init__(self,min_size = 600 , max_size=1000):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self,data):
        img,target = data
        _,H,W = img.shape
        img = preprocess(img,self.min_size,self.max_size)
        _,o_H,o_W = img.shape
        scale = o_H/H
        target["boxes"] = resize_bbox(target["boxes"],(H,W),(o_H,o_W))
        return img,target

def resize_bbox(bbox,in_size,out_size):
    y_scale = float(out_size[0])/ in_size[0]
    x_scale = float(out_size[1])/ in_size[1]
    bbox[:,0] = y_scale * bbox[:,0]
    bbox[:,1] = x_scale * bbox[:,1]
    bbox[:,2] = y_scale * bbox[:,2]
    bbox[:,3] = x_scale * bbox[:,3]
    return bbox



def preprocess(img,min_size,max_size):
    C,H,W = img.shape
    scale1 = min_size/min(H,W)
    scale2 = max_size/max(H,W)
    scale = min(scale1,scale2)
    return transforms.Resize((round(H*scale),round(W*scale)))(img)
