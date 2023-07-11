import sys
import os
sys.path.append(os.path.realpath('.')+'/[third_party]')
third_party_dir = os.path.dirname(os.path.realpath(__file__)) + '/[third_party]'
sys.path.append(third_party_dir)
print('third_party_dir :', third_party_dir)
print('')

## setup path ##################
solution_dir = os.path.dirname(os.path.realpath(__file__))[:-4]
TRAIN_DIR    = f'{solution_dir}/data/vesuvius-challenge-ink-detection'
PRETRAIN_DIR = f'{solution_dir}/data/pretrain'
OUT_DIR      = f'{solution_dir}/results'

print('TRAIN_DIR   :', TRAIN_DIR)
print('PRETRAIN_DIR:', PRETRAIN_DIR)
print('OUT_DIR     :', OUT_DIR)
print('')


################################

import os
#os.environ['QT_DEBUG_PLUGINS']='1'
# mark as root sources in pycharm
from my_lib.other import *
from my_lib.draw import *
from my_lib.file import *
from my_lib.net.rate import *

###################################################################################################3

import math
import numpy as np
import random
import time

import pandas as pd
import json
import zipfile
from shutil import copyfile

from timeit import default_timer as timer
import itertools
import collections
from collections import OrderedDict
from collections import defaultdict
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
print('matplotlib.get_backend : ', matplotlib.get_backend())
#print(matplotlib.__version__)


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

#---
import cv2

def set_environment(seed = int(time.time())):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    #---------------
    common_string = '@%s:  \n' % os.path.basename(__file__)

    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True
    
    # seed
    common_string += '\t\tseed = %d\n'%seed
    common_string += '\n'
    
    # pytorch
    common_string += '\tpytorch\n'
    common_string += '\t\ttorch.__version__              = %s\n'%torch.__version__
    common_string += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    common_string += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    try:
        common_string += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
    except Exception:
        common_string += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
      
    common_string += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    common_string += '\t\ttorch.cuda.get_device_properties() = %s\n' % str(torch.cuda.get_device_properties(0))[21:]
    common_string += '\n'

    return common_string



if __name__ == '__main__':
    common_string = set_environment()
    print (common_string)
    
  
