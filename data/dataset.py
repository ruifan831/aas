from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
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
            self.ids = f.readlines()
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        idx = self.ids[idx]
        img_path = os.path.join(self.root_dir,"JPEGImages",idx+".jpg")
        img = Image.open(img_path).convert("RGB")
        anno = ET.parse(os.path.join(self.root_dir,"Annotations",id_+".xml"))
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
            img,target = self.transform(img,target)
        return img, target
