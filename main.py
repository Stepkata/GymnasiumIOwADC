import gymnasium as gym
from models import ModelType
from policy_network import Policy_Network

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

plt.rcParams["figure.figsize"] = (10, 5)


frozen_lake_env = gym.make(
    ModelType.FROZEN_LAKE,
    desc=None,
    map_name='4x4',
    is_slippery=True
)

