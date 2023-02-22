import os
import sys
import glob
import time
import argparse
import cv2
import torch
import yaml

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

sys.path.insert(0,'training/')

from pprint import pprint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from midas_loss import *
from training.src.plmodel import DepthPLModel
from training.src.utils import load_config, print_config
from omegaconf import OmegaConf, DictConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path to config file', default=None)
    return parser.parse_args()

class CustomDataset():
    def __init__(self, config):
        self.imgs_path = config['DATASET']['TRAIN']
        self.labels_path = config['DATASET']['TEST']

        self.images_list = [os.path.basename(x) for x in glob.glob(self.imgs_path + "*")]
        self.labels_list = [os.path.basename(x) for x in glob.glob(self.labels_path + "*")]
        self.find_data_intersections()

        assert len(self.images_list) == len(self.labels_list)
        
        self.img_dim = config['DATA']['SCALE_SIZE']
    
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
    args = parse_args()
    config_path = args.config_path
    config = load_config(config_path)    

    train_dataset = CustomDataset(config)
    valid_dataset = CustomDataset(config)
    test_dataset = CustomDataset(config)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    batch_size = config['SOLVER']['BATCHSIZE']

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    
    model = DepthPLModel(config)
    print(model)

    logger = TensorBoardLogger("tb_logs", name="mobilenetv3_psp")
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.SOLVER.num_epochs,
        log_every_n_steps=config.SOLVER.SAVE_INTERVAL,
        logger=logger
    )
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


    #ckp = "/home/yegor/repos/MiDaS/tb_logs/mobilenetv3_psp/version_5/checkpoints/epoch\=69-step\=980.ckpt"
    ##model = MiDaSModel.load_from_checkpoint(checkpoint_path=ckp)
    #imgs_path = "depth_dataset/images/"
    ##imgs_path = "input/"
    #imgs_list = [x for x in glob.glob(imgs_path + "*")]
    #
    #for img_name in imgs_list:
    #    print('i name', img_name)
    #    img = cv2.imread(img_name)
    #    img_dim = (256, 256)
    #    img = cv2.resize(img, img_dim)

    #    img_tensor = torch.from_numpy(img)
    #    img_tensor = img_tensor.permute(2, 0, 1)
    #    output = model(img_tensor) * 255
    #    cv2.imwrite(f'training_output/{os.path.basename(img_name)}', output.detach().numpy().reshape(256, 256, 1))
    

    #valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    #pprint(valid_metrics)
    # 
    #test_res = trainer.test(dataloaders=test_dataloader, ckpt_path=ckp)
    #pprint(test_res)
