
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ddpg_model import (Actor, Critic)
# from memory import  SequentialMemory

from utils.ddpg_utils import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class DDPG(object):
    def __init__(self, nb_states, nb_actions, **kwargs):

        if kwargs["seed"] > 0:
            self.seed(kwargs["seed"])

        self.nb_states = nb_states
        self.nb_actions= nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':kwargs["hidden1"],
            'hidden2':kwargs["hidden2"],
            'init_w':kwargs["init_w"]
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=kwargs["prate"])


        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=kwargs["rate"])

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = SequentialMemory(limit=kwargs["rmsize"], window_length=kwargs["window_length"])
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=kwargs["ou_theta"], mu=kwargs["ou_mu"],
                                                       sigma=kwargs["ou_sigma"])

        # Hyper-parameters
        self.batch_size = kwargs["bsize"]
        self.tau = kwargs["tau"]
        self.discount = kwargs["discount"]
        self.depsilon = 1.0 / kwargs["epsilon"]

        #
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)


        # reward_mean = np.mean(reward_batch)
        # reward_std = np.std(reward_batch)
        # reward_batch = (reward_batch - reward_mean)/reward_std

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            next_state_batch.cuda(),
            self.actor_target(next_state_batch.cuda()),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ state_batch.cuda(), to_tensor(action_batch) ])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            state_batch.cuda(),
            self.actor(state_batch.cuda())
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss,policy_loss


    def set_lr(self,prate, rate):

        self.critic_optim = Adam(self.critic.parameters(), lr=rate)
        self.actor_optim = Adam(self.actor.parameters(), lr=prate)

    def eval(self):
        self.is_training = False
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        self.epsilon -= self.depsilon
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(s_t)
        )
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output, step = "origin"):
        if output is None: return

        self.actor.load_state_dict(
            torch.load(f'{output}/actor_{step}.pkl')
        )

        self.critic.load_state_dict(
            torch.load(f'{output}/critic_{step}.pkl')
        )


    def save_model(self,output, step="origin"):
        torch.save(
            self.actor.state_dict(),
            f'{output}/actor_{step}.pkl'
        )
        torch.save(
            self.critic.state_dict(),
            f'{output}/critic_{step}.pkl'
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)


