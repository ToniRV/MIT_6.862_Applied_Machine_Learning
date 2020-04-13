import numpy as np
import itertools
import math as m
from matplotlib import pyplot as plt

import torch
from torch.nn import Linear, ReLU, Softmax, Sequential, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from utils_hw7 import (plot_heat, get_data_loader, model_fit,
                        model_evaluate, run_pytorch, run_pytorch_2d)


######################################################################
# Problem 3 - 2D data
######################################################################

def archs(classes):
    return {0: [Linear(in_features=2, out_features=classes, bias=True),
             Softmax(dim=-1)],
            
            1: [Linear(in_features=2, out_features=10, bias=True),
             ReLU(),
             Linear(in_features=10, out_features=classes, bias=True),
             Softmax(dim=-1)],

            2: [Linear(in_features=2, out_features=100, bias=True),
             ReLU(),
             Linear(in_features=100, out_features=classes, bias=True),
             Softmax(dim=-1)],
            
            3: [Linear(in_features=2, out_features=10, bias=True),
             ReLU(),
             Linear(in_features=10, out_features=10, bias=True),
             ReLU(),
             Linear(in_features=10, out_features=classes, bias=True),
             Softmax(dim=-1)],

            4: [Linear(in_features=2, out_features=100, bias=True),
             ReLU(),
             Linear(in_features=100, out_features=100, bias=True),
             ReLU(),
             Linear(in_features=100, out_features=classes, bias=True),
             Softmax(dim=-1)]
           }



"""## 3G)"""

points = np.array([[-1,0], [1,0], [0,-11], [0,1], [-1,-1], [-1,1], [1,1], [1,-1]])

deterministic = False
if deterministic:
  torch.manual_seed(10)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(10)

"""
HERE YOUR CODE!!
"""

pass

"""# Playground"""
if __name__ == '__main__':
    _ = run_pytorch_2d("1", archs(2)[0], 10, display=True, verbose=True, trials=1)

    #_ = run_pytorch_2d("3class", archs(3)[3], 10, split=.5, display=False, verbose=False, trials=20)

"""
HERE YOUR Experiments!!
"""
