from os.path import join
from os import listdir

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import pdb

from params import ROTNIST_ROOT

# ROTNIST_ROOT = "/home/sungwon/ws/dnqnet/data/RotNIST/data"
import argparse


import pdb

class RotNISTDataset(Dataset):
	def __init__(self, args, mode, transforms=None):
		self.img_dir = self._getImgdir(mode)
		self.img_path_list = self._getImgpathlist(args, mode)
		self.label_list = torch.tensor(self._getLabellist())
		self.transforms = transforms
		self.args = args

	def __len__(self):
		return len(self.img_path_list)

	def __getitem__(self, idx):
		img = Image.open(self.img_path_list[idx])

		if self.transforms is not None:
			img = self.transforms(img)

		return img, self.label_list[idx]

	def _getImgdir(self, mode):
		if mode == 'val':
			mode = 'train'
		return join(ROTNIST_ROOT, mode+'-images')

	def _getImgpathlist(self, args, mode):
		imglist = [join(self.img_dir, f) for f in listdir(self.img_dir) if f.split(".")[-1] == 'jpg']
		l = int(len(imglist)*args.train_val_ratio)

		if mode == 'train':
			return imglist[:l]

		elif mode == 'val':
			return imglist[l:]

		else:
			return imglist


	def _getLabellist(self):
		return [int(f.split("/")[-1].split("_")[-1].split(".")[0]) for f in listdir(self.img_dir) if f.split(".")[-1] == 'jpg']

