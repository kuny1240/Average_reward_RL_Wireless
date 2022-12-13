import numpy as np

import torch
import torch.optim as optim
import copy
from torch.optim import SGD, Adam
from memory import RecurrentExperienceReplayMemory
import pdb


class QLearner:
    def __init__(self, mac, device, batch_size, num_feats, est_type):
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        self.memory = RecurrentExperienceReplayMemory(capacity=1000, sequence_length=1)
        self.seq_len = 1
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        self.learn_start = 200
        self.update_freq = 64
        self.params = list(mac.parameters())
        self.optimiser = Adam(params=self.params, lr=0.01)
        self.gamma = 0.9
        self.normalization_const = 10.
        self.grad_norm_clip = 500.
        self.eva_avg_reward = 0
        self.tar_avg_reward = 0
        self.est_type = est_type

    def remember(self, state, action, reward, state_1):
        self.memory.push((state, action, reward,state_1))

    def set_sgd(self, lr):
        self.optimiser = Adam(params=self.params, lr=lr)

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1, self.num_feats)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(list(batch_action), device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(list(batch_reward), device=self.device, dtype=torch.float).squeeze().view(-1, 1)
#         batch_avg_reward = torch.tensor(list(batch_avg_reward), device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device,dtype=torch.float)


        return batch_state, batch_action, batch_reward, batch_next_state

    def train(self, s, a, r, s_, frame = 0):
        # batch["obs"].shape=[batch, t, obs]

        self.remember(s,a,r,s_)

#         self.target_mac.init_hidden(self.batch_size)

#         self.mac.init_hidden(self.batch_size)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None, None


#         breakpoint()
#         norm = torch.nn.Functional.normalize()
        batch = self.prep_minibatch()
        mac_out = []
        tar_mac_out = []

        batch_state, batch_action, batch_reward, batch_next_state = batch
        batch_state = torch.reshape(batch_state, (self.batch_size,self.seq_len,-1))

        for i in range(self.seq_len):
            cur_state = batch_state[:,i]
#             batch_out, hidden_state = self.mac.forward(cur_state)
            batch_out = self.mac.forward(cur_state)
            tar_batch_out = self.target_mac.forward(cur_state)
            tar_mac_out.append(tar_batch_out)
            mac_out.append(batch_out)

        mac_out = torch.stack(mac_out, dim=1)
        mac_out = torch.reshape(mac_out, (self.batch_size*self.seq_len, -1))
        tar_mac_out = torch.stack(tar_mac_out, dim=1)
        tar_mac_out = torch.reshape(tar_mac_out, (self.batch_size*self.seq_len, -1))
        # mac_out, hidden_states = self.mac.forward(batch_state)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = mac_out.gather(1, batch_action).squeeze()  # Remove the last dim
        chosen_action_qvals = torch.nn.functional.normalize(chosen_action_qvals,dim=0)
#         chosen_action_qvals = norm(chosen_action_qvals)
        target_chosen_qvals = tar_mac_out.gather(1, batch_action).squeeze()
#         target_chosen_qvals = norm(target_chosen_qvals)

        # Calculate the Q-Values necessary for the target

        target_mac_out = []
        batch_next_state = torch.reshape(batch_next_state, (self.batch_size, self.seq_len, -1))

        for i in range(self.seq_len):
            cur_state = batch_next_state[:,i]
#             batch_next_out, hidden_state = self.mac.forward(cur_state)
            batch_next_out = self.target_mac.forward(cur_state)
            target_mac_out.append(batch_next_out)

        target_mac_out = torch.stack(target_mac_out, dim=1)
        target_mac_out = torch.reshape(target_mac_out, (self.batch_size * self.seq_len, -1))

        # We don't need the first timesteps Q-Value estimate for calculating targets
        # Max over target Q-Values

        # Get actions that maximise live Q (for double q-learning)
        target_max_qvals = target_mac_out.max(dim=1)[0]
#         target_max_qvals = norm(target_max_qvals)
        # target_max_qvals = torch.gather(target_mac_out, 2, cur_max_actions).squeeze(3)

        # Calculate 1-step Q-Learning targets
        batch_reward = torch.flatten(batch_reward)
#         batch_reward = norm(batch_reward)
#         avg_reward = torch.mean(batch_reward)
        
        
        if self.est_type == "discounted":
            targets = batch_reward + self.gamma * target_max_qvals
            targets = torch.nn.functional.normalize(targets,dim=0)
        else:
            targets = batch_reward - self.tar_avg_reward + target_max_qvals
            self.eva_avg_reward += 0.01 * ( torch.mean(batch_reward - self.eva_avg_reward + target_max_qvals - target_chosen_qvals)).item()
        

        # Td-error
#         pdb.set_trace()
#         td_error = chosen_action_qvals - targets.detach()
#         bad_id = torch.argsort(torch.abs(td_error))

        # Normal L2 loss, take mean over actual data
#         loss = (td_error ** 2).mean()  # + self.normalization_const * avg_difference

        # Optimise
#         loss_func = torch.nn.HuberLoss()
        loss_func = torch.nn.MSELoss(reduction = "sum")
        loss = loss_func(targets.detach(), chosen_action_qvals)
        if loss > 1000:
            pdb.set_trace()
        self.optimiser.zero_grad()
        loss.backward()
        # print(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        self.optimiser.step()
        return loss, grad_norm

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.tar_avg_reward = copy.deepcopy(self.eva_avg_reward)



    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()