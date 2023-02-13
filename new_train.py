import os
import time
import cv2
import torch
import glob
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import sys
sys.path.insert(0,'training/')

from pprint import pprint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from midas_loss import *
from training.src.plmodel import DepthPLModel
from omegaconf import OmegaConf, DictConfig

class CustomDataset():
    def __init__(self):
        self.imgs_path = "depth_dataset/images/"
        self.labels_path = "depth_dataset/labels/"

        self.images_list = [os.path.basename(x) for x in glob.glob(self.imgs_path + "*")]
        self.labels_list = [os.path.basename(x) for x in glob.glob(self.labels_path + "*")]
        self.find_data_intersections()

        assert len(self.images_list) == len(self.labels_list)
        
        self.img_dim = (256, 256)
    
    def __len__(self):
        return len(self.images_list)

    def find_data_intersections(self):
        self.intersection_list = [value for value in self.images_list if value in self.labels_list]
        self.images_list = list(sorted([self.imgs_path + value for value in self.intersection_list]))
        self.labels_list = list(sorted([self.labels_path + value for value in self.intersection_list]))

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        label_path = self.labels_list[idx]

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, self.img_dim).reshape(self.img_dim[0], self.img_dim[1], 1)

        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)

        label_tensor = torch.from_numpy(label) / 255
        label_tensor = label_tensor.permute(2, 0, 1)

        return {'image': img_tensor.float(), 'depth': label_tensor.float()}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # init train, val, test sets
    train_dataset = CustomDataset()
    valid_dataset = CustomDataset()
    test_dataset = CustomDataset()
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)
    
    model = DepthPLModel()
    print(model)

    logger = TensorBoardLogger("tb_logs", name="mobilenetv3_psp")
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=50,
        log_every_n_steps=4,
        logger=logger
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
