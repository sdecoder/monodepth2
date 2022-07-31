from collections import defaultdict
from glob import glob

import torch
from pathlib import Path
import random
from pathlib import Path
import importlib
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader

from IPython.display import display
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from scipy.spatial.distance import euclidean
from imageio import imread
from skimage.transform import resize

# some of blocks below are not used.

# Data manipulation
import numpy as np
import pandas as pd

# Data visualisation
import matplotlib.pyplot as plt

# Fastai
# from fastai.vision import *
# from fastai.vision.models import *

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import *
from torch.autograd import Variable
# import pretrainedmodels

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from alive_progress import alive_bar
from torch import nn, optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

import argparse

import onnx
import tensorrt as trt
import os
import torch
import torchvision
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum

pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable




class Calibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, training_loader, cache_file, element_bytes, batch_size=16, ):
    # Whenever you specify a custom constructor for a TensorRT class,
    # you MUST call the constructor of the parent explicitly.
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = cache_file
    self.data_provider = training_loader
    self.batch_size = batch_size
    self.current_index = 0

    # we assume single element is 4 byte
    mem_size = element_bytes * batch_size
    print(f'[trace] allocated mem_size: {mem_size}')
    self.device_input0 = cuda.mem_alloc(mem_size)
    self.device_input1 = cuda.mem_alloc(mem_size)

  def get_batch_size(self):

    return self.batch_size

  # TensorRT passes along the names of the engine bindings to the get_batch function.
  # You don't necessarily have to use them, but they can be useful to understand the order of
  # the inputs. The bindings list is expected to have the same ordering as 'names'.
  def get_batch(self, names):

    max_data_item = len(self.data_provider.dataset)
    if self.current_index + self.batch_size > max_data_item:
      return None

    _imgs0, _imgs1, labels = next(iter(self.data_provider))
    _elements0 = _imgs0.ravel().numpy()
    _elements1 = _imgs1.ravel().numpy()

    cuda.memcpy_htod(self.device_input0, _elements0)
    cuda.memcpy_htod(self.device_input1, _elements1)
    self.current_index += self.batch_size
    return [self.device_input0, self.device_input1]

  def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    print(f'[trace] Calibrator: read_calibration_cache: {self.cache_file}')
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()

  def write_calibration_cache(self, cache):
    print(f'[trace] Calibrator: write_calibration_cache: {cache}')
    with open(self.cache_file, "wb") as f:
      f.write(cache)


def GiB(val):
  return val * 1 << 30


class trainingDataset(Dataset):  # Get two images and whether they are related.

  def __init__(self, imageFolderDataset, relationships, transform=None):

    self.imageFolderDataset = imageFolderDataset
    self.relationships = relationships  # choose either train or val dataset to use
    self.transform = transform

  def __getitem__(self, index):
    img0_info = self.relationships[index][0]
    # for each relationship in train_relationships.csv, the first img comes from first row, and the second is either specially choosed related person or randomly choosed non-related person
    img0_path = glob("../data/train/" + img0_info + "/*.jpg")
    img0_path = random.choice(img0_path)

    cand_relationships = [x for x in self.relationships if
                          x[0] == img0_info or x[1] == img0_info]  # found all candidates related to person in img0
    if cand_relationships == []:  # in case no relationship is mensioned. But it is useless here because I choose the first person line by line.
      should_get_same_class = 0
    else:
      should_get_same_class = random.randint(0, 1)

    if should_get_same_class == 1:  # 1 means related, and 0 means non-related.
      img1_info = random.choice(cand_relationships)  # choose the second person from related relationships
      if img1_info[0] != img0_info:
        img1_info = img1_info[0]
      else:
        img1_info = img1_info[1]
      img1_path = glob("../data/train/" + img1_info + "/*.jpg")  # randomly choose a img of this person
      img1_path = random.choice(img1_path)
    else:  # 0 means non-related
      randChoose = True  # in case the chosen person is related to first person
      while randChoose:
        img1_path = random.choice(self.imageFolderDataset.imgs)[0]
        img1_info = img1_path.split("/")[-3] + "/" + img1_path.split("/")[-2]
        randChoose = False
        for x in cand_relationships:  # if so, randomly choose another person
          if x[0] == img1_info or x[1] == img1_info:
            randChoose = True
            break

    img0 = Image.open(img0_path)
    img1 = Image.open(img1_path)

    if self.transform is not None:  # I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
      img0 = self.transform(img0)
      img1 = self.transform(img1)

    return img0, img1, should_get_same_class  # the returned data from dataloader is img=[batch_size,channels,width,length], should_get_same_class=[batch_size,label]

  def __len__(self):
    return len(self.relationships)  # essential for choose the num of data in one epoch

val_famillies = "F09"
IMG_SIZE = 100
LOADER_WORER_NUMBER = 1
BATCH_SIZE = 64

def prepare_train_data():
  import sys
  sys.path.append('..')
  from options import MonodepthOptions

  options = MonodepthOptions()
  opt = options.parse()
  print(f'[trace] does the options class loaded?')

  sys.path.append('../..')
  from datasets import KITTIRAWDataset, KITTIDepthDataset, KITTIOdomDataset
  datasets_dict = {"kitti": KITTIRAWDataset,
                   "kitti_odom": KITTIOdomDataset}
  dataset = datasets_dict[opt.dataset]
  current_path_name = os.path.dirname(__file__)
  from pathlib import Path
  path = Path(os.getcwd())
  project_root = path.parent.parent.absolute()
  dataset_path = path.joinpath(project_root, 'datasets')

  fpath = os.path.join(dataset_path.absolute(), "splits", opt.split, "{}_files.txt")
  img_ext = '.png' if opt.png else '.jpg'
  def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
      lines = f.read().splitlines()
    return lines

  train_filenames = readlines(fpath.format("train"))
  train_data_dir = os.path.join(project_root, 'datasets/splits/')
  train_dataset = dataset(
    train_data_dir, train_filenames, opt.height, opt.width,
    opt.frame_ids, 4, is_train=True, img_ext=img_ext)

  train_loader = DataLoader(
    train_dataset, opt.batch_size, True,
    num_workers=opt.num_workers, pin_memory=True, drop_last=True)
  return train_loader
