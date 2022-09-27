import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import RMSprop, SGD, Adam


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, states):
        return th.sum(agent_qs, dim=2, keepdim=True)

class QMixer(nn.Module):
    def __init__(self, n_agents = 4, state_shape = 30, mixing_embed_dim = 10):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_shape #int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0) # batch_size
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class QLearner:
    def __init__(self, mac, n_ues=10, info_reg=10.):
        self.mac = mac
        self.target_mac = copy.deepcopy(mac)
        
        self.params = list(mac.parameters())

        #self.mixer = VDNMixer()
        self.mixer = QMixer(n_agents=self.mac.n_agents, state_shape=3*n_ues)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.params += list(self.mac.env_blender.parameters())
        #self.optimiser = RMSprop(params=self.params, lr=0.1, alpha=0.99, eps=0.00001)
        self.optimiser = SGD(params=self.params, lr=0.01, momentum = 0.9)

        self.gamma = 0.0 #0.98
        self.normalization_const = info_reg
        self.grad_norm_clip = 500.

    def set_sgd(self, lr, mmt=0.0):
        self.optimiser = SGD(params=self.params, lr=lr, momentum = mmt)

    def set_adam(self, lr):
        self.optimiser = Adam(params=self.params, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    def set_rms(self, lr):
        self.optimiser = RMSprop(params=self.params, lr=lr, alpha=0.99, eps=0.00001)

    def train(self, batch, commu = False):
        # batch["obs"].shape=[batch, t, obs]

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1].unsqueeze(-1)
        states = batch["state"]
        batch = batch["obs"]
        mask = batch[:, :, :, :, 0]
        batch_size = batch.shape[0]
    
        # Calculate estimated Q-Values
        mac_out = []
        difference_out = []
        zero_const = th.zeros([batch_size, self.mac.n_agents, 1]).cuda()
        self.mac.init_hidden(batch_size)
        for t in range(batch.shape[1]):
            agent_local_outputs, hidden_states = self.mac.forward(batch[:, t])
            agent_outs = agent_local_outputs
            if commu:
                dummy = th.stack([self.mac.env_blender(hidden_states[:,i,:].view(batch_size,-1)) for i in range(self.mac.n_agents)], dim = 1)
                
                message_sum = dummy.sum(1, keepdim=True)
                agent_message = (message_sum - dummy) / (self.mac.n_agents - 1.0)
        
                agent_global_outputs = th.gather(agent_message, -1, mask[:, t].long())
                agent_global_outputs = th.cat([agent_global_outputs, zero_const.detach()], -1)
                agent_outs += agent_global_outputs
                difference = agent_global_outputs 
                difference_out.append(difference)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over 
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        if commu:
            difference_out = th.stack(difference_out, dim=1)  # Concat over time
            difference_out = th.std(difference_out,dim = 3).sum()
            avg_difference = difference_out/((agent_outs.shape[0]*agent_outs.shape[1]*agent_outs.shape[2]*batch.shape[1]))

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch_size)
        for t in range(batch.shape[1]):
            batch[:, t]
            target_agent_local_outputs, target_hidden_states = self.target_mac.forward(batch[:, t])
            target_agent_outs = target_agent_local_outputs

            if commu:
                dummy = th.stack([self.mac.env_blender(target_hidden_states[:,i,:].view(batch_size,-1)) for i in range(self.mac.n_agents)], dim = 1)
                
                message_sum = dummy.sum(1, keepdim=True)
                agent_message = -(dummy - message_sum) / (self.mac.n_agents - 1.0)

                target_agent_global_outputs = th.gather(agent_message, -1, mask[:, t].long())
                target_agent_global_outputs = th.cat([target_agent_global_outputs, zero_const.detach()], -1)
                target_agent_outs += target_agent_global_outputs
            target_mac_out.append(target_agent_outs)
          
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Max over target Q-Values
        
        # Get actions that maximise live Q (for double q-learning)
        t_mac_out = mac_out.clone()
        cur_max_actions = t_mac_out[:, 1:].max(dim=3, keepdim=True)[1]
        target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        #target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, states[:, :-1]).squeeze(-1)
        target_max_qvals = self.target_mixer(target_max_qvals, states[:, 1:]).squeeze(-1)
        #target_max_qvals[]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * target_max_qvals

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).mean()
        if commu:
            loss += self.normalization_const * avg_difference

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        #print(loss)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.grad_norm_clip)
        #print(grad_norm)
        self.optimiser.step()
        if commu:
            return loss, grad_norm, avg_difference
        else:
            return loss, grad_norm

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()