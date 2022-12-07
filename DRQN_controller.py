from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
# from ENV import DHenv as Env
# from memory import RecurrentExperienceReplayMemory
from DRQN_agent import DRQN as agent

import numpy as np
import math
import copy
import matplotlib.pyplot as plt


class DRQN_Agent:
    def __init__(self, n_actions=256, input_shape=60):
        self.n_actions = n_actions
        self.input_shape = input_shape
        self._build_agents(input_shape)
        self.hidden_states = None
        self.cuda_flag = False

    def get_action(self, s, eps=0.1):
#         breakpoint()
        s = s.cuda()
#         assert s.device == "cpu"
        if s.device == "cpu":
            breakpoint()
        with torch.no_grad():
            if np.random.random() >= eps:
                q_value, hidden = self.forward(s)
                a = q_value.argmax(dim=-1)
                return a.item(), q_value
            else:
                q_value = torch.zeros((self.n_actions,)).cuda()
                act = np.random.randint(0, self.n_actions)
                q_value[act] = 1
                return np.random.randint(0, self.n_actions), q_value
            
            

    def forward(self, ep_batch):
        # ep_batch.shape == [batch, n_agents, limited_n_ues, 4(No., x, y, patience)]
        batch = ep_batch.shape[0]
        agent_inputs = ep_batch.view(batch, -1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs.view(batch, -1), self.hidden_states.view(batch, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = Variable(torch.zeros(1, batch_size, self.agent.gru_size).float())  # bav
        if self.cuda_flag:
            self.hidden_states = self.hidden_states.cuda()

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()
        self.cuda_flag = True

    def _build_agents(self, input_shape):
        self.agent = agent(input_shape, self.n_actions, gru_size=64)

    def save(self, pth):
        torch.save(
            self.agent.state_dict(), f'{pth}.pkl'
            )

    def load(self, pth):
            self.agent.load_state_dict(
                torch.load(f'{pth}.pkl')
            )


    def load_state(self, m):
        self.agent.load_state_dict(m.agent.state_dict())