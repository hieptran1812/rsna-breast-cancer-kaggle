import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class RsnaDataset(Dataset):
    def __init__(self, df, cfg, aug):
        self.cfg = cfg
        self.df = df.copy()
        self.labels = self.df[self.cfg.classes].values
        self.aug = aug
        self.data_folder = cfg.data_folder
        self.df['fns'] = self.df['patient_id'].astype(str) + '_' + self.df['image_id'].astype(str) + '.png'
        self.fns = self.df['fns'].astype(str).values
        self.data_folder = cfg.data_folder

    def __len__(self):
        return len(self.fns)

    def normalize_img(self, img):
        img = img / 255
        return img

    def augment(self, img):
        img = img.astype(np.float32)
        transformed = self.aug(image=img)
        trans_img = transformed['image']
        return trans_img

    def load_one(self, idx):
        path = self.data_folder + self.fns[idx]
        # print('path', path)
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            shape = img.shape
            if len(img.shape) == 2:
                img = img[:, :, None]
        except Exception as e:
            print(e)
        return img

    def __getitem__(self, idx):
        label = self.labels[idx]
        # print('label', label)
        img = self.load_one(idx)
        if self.aug:
            img = self.augment(img)
        img = self.normalize_img(img)
        torch_img = torch.tensor(img).float().permute(2, 0, 1)
        feature_dict = {
                'input': torch_img,
                'target': torch.tensor(label)
                }
        return feature_dict
        
