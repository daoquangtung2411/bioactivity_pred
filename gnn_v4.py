import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_max_pool, global_mean_pool, global_add_pool
from hyperopt import hp, tpe, fmin, Trials, space_eval
from hyperopt.pyll import scope
from helper.load_dataset import load_bace_classification
from tabulate import tabulate
from helper.preprocess import split_train_valid_test, generate_graph_dataset
from helper.trainer import fit_model, evaluate_test, final_fit_model, final_evaluate
from helper.graphfeat import StructureEncoderV4
from helper.cal_metrics import classification_metrics