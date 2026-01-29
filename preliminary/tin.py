from torch.utils.data import DataLoader, Dataset
import os
import torch
import glob
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


class tinyimagenet(Dataset):
    def __init__(self, set, mappings_dict):
        super(tinyimagenet, self).__init__()
        self.image_paths = []
        self.labels = []
        self.label_ids = []
        # run imagenet_utils/tinyimagenet.sh to download tinyimagenet data
        if set == 'train':
            path = 'preliminary/data/tiny-imagenet-200/train/'
        elif set == 'val':
            path = 'preliminary/data/tiny-imagenet-200/val/'
        else:
            raise NotImplementedError
        for idx, label in enumerate(mappings_dict.keys()):
            temp_path = path + f"{mappings_dict[label]}"
            self.image_paths.extend(glob.glob(os.path.join(temp_path,'*.JPEG')))
            self.labels.extend(len(glob.glob(os.path.join(temp_path,'*.JPEG')))*[label])
            self.label_ids.extend(len(glob.glob(os.path.join(temp_path,'*.JPEG')))*[idx])
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        self.classes = list(mappings_dict.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            x = self.transform(x)
            y = self.label_ids[index]
        return x, y

def tinyimage_loader(set='train'):
     
    f = open('preliminary/data/tinyimagenet_labels_to_ids.txt', 'r')
    #f = open('../tinyimagenet_ids_to_label.txt', 'r')
    tinyimg_label2folder = f.readlines()
    mappings_dict = {}
    for line in tinyimg_label2folder:
        label, class_id = line[:-1].split(' ')[0], line[:-1].split(' ')[1]
        label = label.replace("_", " ")
        mappings_dict[label] = class_id

    dataset = tinyimagenet(set, mappings_dict)
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=4, shuffle=True)
    return loader

loader = tinyimage_loader('val')