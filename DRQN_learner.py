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
        self.memory = RecurrentExperienceReplayMemory(capacity=1000, sequence_length=batch_size)
        self.seq_len = batch_size
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        self.learn_start = 200
        self.update_freq = 64
        self.params = list(mac.parameters())
        self.optimiser = Adam(params=self.params, lr=0.01)
        self.gamma = 0.98
        self.normalization_const = 10.
        self.grad_norm_clip = 500.
        self.avg_reward = 0
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

        self.target_mac.init_hidden(self.batch_size)

        self.mac.init_hidden(self.batch_size)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None, None


#         breakpoint()
        batch = self.prep_minibatch()
        mac_out = []

        batch_state, batch_action, batch_reward, batch_next_state = batch
        batch_state = torch.reshape(batch_state, (self.batch_size,self.seq_len,-1))

        for i in range(self.batch_size):
            cur_state = batch_state[:,i]
            batch_out, hidden_state = self.mac.forward(cur_state)
            mac_out.append(batch_out)

        mac_out = torch.stack(mac_out, dim=1)
        mac_out = torch.reshape(mac_out, (self.batch_size*self.seq_len, -1))
        # mac_out, hidden_states = self.mac.forward(batch_state)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = mac_out.gather(1, batch_action).squeeze()  # Remove the last dim

        # Calculate the Q-Values necessary for the target

        target_mac_out = []
        batch_next_state = torch.reshape(batch_state, (self.batch_size, self.seq_len, -1))

        for i in range(self.batch_size):
            cur_state = batch_next_state[:,i]
            batch_next_out, hidden_state = self.mac.forward(cur_state)
            target_mac_out.append(batch_next_out)

        target_mac_out = torch.stack(target_mac_out, dim=1)
        target_mac_out = torch.reshape(target_mac_out, (self.batch_size * self.seq_len, -1))

        # We don't need the first timesteps Q-Value estimate for calculating targets
        # Max over target Q-Values

        # Get actions that maximise live Q (for double q-learning)
        target_max_qvals = target_mac_out.max(dim=1)[0]
        # target_max_qvals = torch.gather(target_mac_out, 2, cur_max_actions).squeeze(3)

        # Calculate 1-step Q-Learning targets
        batch_reward = torch.flatten(batch_reward)
#         avg_reward = torch.mean(batch_reward)
        
        
        if self.est_type == "discounted":
            targets = batch_reward + self.gamma * target_max_qvals
        else:
            targets = batch_reward - self.avg_reward + target_max_qvals
            self.avg_reward += 0.01 * ( torch.mean(batch_reward - self.avg_reward + target_max_qvals - chosen_action_qvals))
        

        # Td-error
        pdb.set_trace()
        td_error = chosen_action_qvals - targets.detach()
        bad_id = torch.argsort(torch.abs(td_error))

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).mean()  # + self.normalization_const * avg_difference

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        # print(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        self.optimiser.step()
        return loss, grad_norm

    def _update_targets(self):
        self.target_mac.load_state(self.mac)



    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()