from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
# from ENV import DHenv as Env
# from memory import RecurrentExperienceReplayMemory

import numpy as np
import math
import copy
import matplotlib.pyplot as plt

class DRQN(nn.Module):

    def __init__(self, input_shape, num_actions,  gru_size=1024, bidirectional=False):
        super(DRQN, self).__init__()

        self.gru_size = gru_size
        self.fc1 = nn.Linear(input_shape, gru_size)
        self.rnn = nn.GRUCell(gru_size, gru_size)
        self.fc2 = nn.Linear(gru_size, num_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.gru_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h






def get_decay(epi_iter):
    decay = math.pow(0.999, epi_iter)
    if decay < 0.05:
        decay = 0.05
    return decay


def decode_act(act, num_ap, num_act):
    action = []
    for i in range(num_ap):
        action.append(act % num_act)
        act = act // num_act


    return torch.tensor(action)





# if __name__ == '__main__':

#     t_ues = torch.tensor([[2.3219, 35.5963],
#                           [4.9058, 7.3772],
#                           [34.1434, 36.6237],
#                           [30.3456, 22.6366],
#                           [26.9608, 20.9890],
#                           [19.3620, 32.8353],
#                           [12.3929, 39.7321],
#                           [5.9339, 29.5289],
#                           [33.2905, 37.5042],
#                           [9.2728, 15.3940]])

#     max_epi_iter = 200
#     max_MC_iter = 500
#     env = Env(border=torch.Tensor([40, 40]),
#               enbs=torch.Tensor([[10, 10, 10],
#                                  [10, 30, 10],
#                                  [10, 10, 30],
#                                  [10, 30, 30]]),
#               ues=t_ues,
#               noise=0.01
#               )
#     agent = Agent(input_shape=60,N_action=1296,batch_size=64)
#     agent.set_lr(1e-5)
#     agent.cuda()
#     agent.reset_hx()
#     train_curve = []
#     plt.figure()
#     for epi_iter in range(max_epi_iter):
#         random.seed()
#         env.reset()
#         hidden = None
#         losses = []
#         rewards = []
#         for MC_iter in range(max_MC_iter):
#             # env.render()
#             obs = []
#             for j in range(env.n_enbs):
#                 obs.append(env.get_agent_obs(j))
#             obs = torch.stack(obs, dim=0)
#             obs = torch.flatten(obs)
#             action, hidden = agent.get_action(torch.reshape(obs,(1,60)).cuda(),hidden, get_decay(epi_iter))
#             act = decode_act(action,4)
#             s_t1,r = env.step(act)
#             rewards.append(r)
#             s_t1 = torch.flatten(s_t1)
#             loss = agent.update(obs.numpy(), action, r.item(), s_t1.numpy(), epi_iter * max_MC_iter + MC_iter)
#             if loss is not None:
#                 losses.append(loss)
#         print('Episode', epi_iter, 'reward', rewards[-1], "loss", np.mean(losses), "epsilon", get_decay(epi_iter))
#         train_curve.append(r)
#         if epi_iter % 10 == 0:
#             plt.plot(train_curve)
#             plt.show()



