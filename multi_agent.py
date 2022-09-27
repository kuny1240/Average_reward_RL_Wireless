# -*- Coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

class multi_agent:
    def __init__(self, env, args):
        self.env = env
        self.agent