import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from scipy import signal
from ddpg import DDPG
from models.baselines import *
import pdb


from utils.comm_utils import *

# def parabolic_antenna(origin, target, pos):
#     costheta = ((target - origin) ** 2 + (pos - origin) ** 2 - (target - pos) ** 2).sum() / \
#                (2 * np.sqrt(((target - origin) ** 2).sum() * ((pos - origin) ** 2).sum()))
#     if costheta > 1:
#         costheta = torch.tensor(1.)
#     if costheta < -1:
#         costheta = torch.tensor(-1.)
#     phi = np.arccos(costheta)
#     gainDb = -min(20, 12*(phi / np.pi * 6)**2)
#     gain = 10. ** (gainDb / 10.)
#     return gain


def state_normalization(states,state_max,state_min, Q):

    stages = np.array(list(range(Q + 1)))/ Q
    normalized_stages = stages - 1/2
    diff = state_max - state_min
    state_stages = stages * diff + state_min

    normalized_states = copy.deepcopy(states)
    for i in range(len(states)):

        pos = np.where(state_stages <= states[i])[0]
        if len(pos) == 0:
            normalized_states[i] = 1/2
        else:
            normalized_states[i] = normalized_stages[pos[-1]]


    return normalized_states




class Env():
    def __init__(self,apNum,ueNum,boarder,random_seed = None):

        self.apNum = apNum
        self.ueNum = ueNum
        self.boarder = boarder
        self.alphaI = 0.05
        self.alphaR = 0.01
        self.visible_ues = 3
        if random_seed != None:
            np.random.seed(random_seed)
            random.seed(random_seed)


        self.reset()

        flag = True

        while flag:
            flag = False
            for i in range(self.apNum):
                if len(self.apUE[i]) == 0:
                    self.reset()
                    flag = True




    def reset(self,random_seed = None):

        if random_seed != None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.time = 0
        self.appos = APgen(self.boarder,self.apNum,35)
        self.uepos = UEgen(self.appos,self.boarder,self.ueNum,10)
        self.dists = getDist(self.appos,self.uepos)
        self.passGain = DSPloss(self.dists)
        self.shadowing = log_norm_shadowing(self.apNum,self.ueNum,7)
        self.apID = np.argmax(self.passGain + self.shadowing,0)
        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        self.apUE = dict()
        for i in range(self.apNum):
            self.apUE[i] = np.where(self.apID == i)[0]
        self.power = np.ones(self.dists.shape) * 0.01
        self.achRate = getAchRate(self.passGain + self.fading + self.shadowing,self.power,-134,self.apID)
        self.weight = 1/self.achRate
        self.max_weight = self.weight.max()
        self.min_weight = self.weight.min()
        self.normalized_weight = state_normalization(self.weight,self.max_weight,self.min_weight,20)
        self.infPower = getInfpower(self.passGain + self.fading + self.shadowing, 0.01 , self.apID)
        self.SINR = getSINR(self.passGain + self.fading + self.shadowing, self.infPower, -134, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
        self.max_SINR = self.SINRdb.max()
        self.min_SINR = self.SINRdb.min()
        self.normalized_SINR = state_normalization(self.SINRdb,self.max_SINR,self.min_SINR,20)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = self.achRate
        self.reward = 0
        self.actions = torch.tensor(np.zeros((self.apNum*2,)),dtype=torch.float32)
        

        #return self.struct_obv()
        return self.global_obv()


    def global_obv(self):



        global_ob = torch.cat((torch.tensor(self.normalized_weight,dtype=torch.float32).view(-1),
                               torch.tensor(self.normalized_SINR,dtype=torch.float32).view(-1)),0)
#                                self.actions,
#                                torch.tensor([self.reward],dtype=torch.float32)),0)

        return global_ob

    def state_dim(self):
        
        return self.global_obv().shape[0]
    
    def action_dim(self):
        
        return 256

    def soft_reset(self):

        self.time = 0
        self.power = np.ones(self.dists.shape) * 0.01
        self.achRate = getAchRate(self.passGain + self.fading + self.shadowing, self.power, -134, self.apID)
        self.weight = 1 / self.achRate
        self.max_weight = self.weight.max()
        self.min_weight = self.weight.min()
        self.normalized_weight = state_normalization(self.weight, self.max_weight, self.min_weight, 20)
        self.infPower = getInfpower(self.passGain + self.fading + self.shadowing, 0.01, self.apID)
        self.SINR = getSINR(self.passGain + self.fading + self.shadowing, self.infPower, -134, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
        self.max_SINR = self.SINRdb.max()
        self.min_SINR = self.SINRdb.min()
        self.normalized_SINR = state_normalization(self.SINRdb, self.max_SINR, self.min_SINR, 20)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = self.achRate
        self.reward = 0
        self.actions = torch.tensor(np.zeros((self.apNum * 2,)), dtype=torch.float32)

    def ap_obv(self, apID):

        ue_id = self.apUE[apID]
        if len(ue_id) < self.visible_ues:
            ue_weights = np.zeros((self.visible_ues,))
            ue_weights[:len(ue_id)] = self.normalized_weight[ue_id]
            ue_SINR = np.zeros((self.visible_ues,)) - 60
            ue_SINR[:len(ue_id)] = self.normalized_SINR[ue_id]
        else:
            ue_pf = self.pf[ue_id]
            ue_id = np.argsort(ue_pf)[::-1][:self.visible_ues]
            ue_weights = self.normalized_weight[ue_id]
            ue_SINR = self.normalized_SINR[ue_id]

#         ue_weights = state_normalization(ue_weights,self.max_weight,self.min_weight,20)
#         ue_SINR = state_normalization(ue_SINR,self.max_SINR,self.min_SINR,20)

        return [ue_weights,ue_SINR]

    def struct_obv(self):
        struct_ob = []
        for i in range(self.apNum):
            struct_ob.append(torch.tensor(self.ap_obv(i), dtype=torch.float32).view(-1))
        struct_ob = torch.stack(struct_ob)
        struct_ob = struct_ob.view(-1)
        return struct_ob

    def step(self, actions):

        ues = []

        # HERE IS A HEURISTIC TRANSFORM
        if len(actions.shape) == 1:
            actions = torch.stack([actions % self.visible_ues, actions // self.visible_ues], dim=1)

        for i in range(self.appos.shape[0]):
            ues.append(self.ap_action(i,actions[int(i),0], 0.01))


        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        instRate = getAchRate(self.passGain + self.fading + self.shadowing,self.power,-134, self.apID)
        mask = np.zeros(instRate.shape)
        for i in range(len(ues)):
            if ues[i] != None:
                mask[ues[i]] = 1
        instRate *= mask
        self.achRate = (1 - self.alphaR) * self.achRate + self.alphaR * instRate
        self.weight = 1/self.achRate
#         if self.weight.max() > self.max_weight:
        self.max_weight = self.weight.max()
#         if self.weight.min() < self.min_weight:
        self.min_weight = self.weight.min()
        self.normalized_weight = state_normalization(self.weight,self.max_weight,self.min_weight,20)
        self.infPower = (1 - self.alphaI) * self.infPower + \
                        self.alphaI * getInfpower(self.passGain + self.fading + self.shadowing, 0.01, self.apID)
        self.SINR = getSINR(self.passGain + self.fading + self.shadowing, self.infPower, -134, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
#         if self.SINRdb.max() > self.max_SINR:
        self.max_SINR = self.SINRdb.max()
#         if self.SINRdb.min() < self.min_SINR:
        self.min_SINR = self.SINRdb.min()
        self.normalized_SINR = state_normalization(self.SINRdb, self.max_SINR, self.min_SINR, 20)
        self.pf = self.weight * np.log2(1 + self.SINR)

        reward = 0
        lambda_rew = 0.8

        for i in range(len(ues)):
            if ues[i] != None:
                reward += self.weight[ues[i]] ** lambda_rew * instRate[ues[i]]

        if reward == 0:
            reward = - np.max(self.pf)

        reward = max(reward, -40)

        self.reward = reward
        self.actions = torch.tensor(np.reshape(actions,(self.apNum*2,)),dtype=torch.float32)

        self.accumulated_rate += instRate
        self.time += 1

        #return self.struct_obv(), torch.tensor(reward, dtype=torch.float32)
        return self.struct_obv(), torch.tensor(reward,dtype=torch.float32)

    def ap_action(self, apID, ueID, power):
        
#         pdb.set_trace()
        ue_id = self.apUE[apID]
        ue_pf = self.pf[ue_id]
        ue_ord = np.argsort(ue_pf)[::-1]
        ue_id = ue_id[ue_ord]


        if ueID >= len(ue_id)  or ueID == self.visible_ues + 1 or power == 0:
            ue = None
            self.power[apID,:] = 0
        else:
            ue = ue_id[int(ueID)]
            self.power[apID,:] = power
            # for i in ue_id:
            #     self.power[apID,i] = power * parabolic_antenna(self.appos[apID,:],\
            #                                                    self.uepos[ue,:],\
            #                                                    self.uepos[i,:]

        return ue

    def get_Rsum(self):
        return np.sum(self.accumulated_rate/self.time)

    def get_R5per(self):

        pos = math.floor(len(self.accumulated_rate) * 0.95) - 1
        sorted_rate = np.sort(self.accumulated_rate)[::-1]

        return sorted_rate[pos]/self.time

    def plot_scheduling(self):
        # use power for size
        s = np.max(self.power,axis=1)
        s = s/np.max(s) * 100
        cm = np.linspace(1, self.apNum, self.apNum)

        plt.figure(figsize=(5,5))
        plt.scatter(self.appos[:,0], self.appos[:,1],marker="x", s = s, c = cm)

        s_rate = (self.achRate)  * 100
        plt.scatter(self.uepos[:,0],self.uepos[:,1],marker="o", s = s_rate, c = self.apID)
        plt.xlim([0,self.boarder])
        plt.ylim([0,self.boarder])
        plt.show()




if __name__ == "__main__":


    apNum = 2
    ueNum = 8
    random_seed = 155
    obs_dim = apNum * 10
    act_dim = apNum * 2

    env = Env(apNum,ueNum,500,random_seed)
    apUE = env.apUE
    Rsum = {"full-reuse":[],"TDM":[],"DDPG":[]}
    R5per = {"full-reuse":[],"TDM":[],"DDPG":[]}
    rewards = {"full-reuse":[],"TDM":[],"DDPG":[]}

    agent = DDPG(obs_dim, act_dim, seed=8, hidden1=400,
                 hidden2=300, bsize=64,
                 rate=0.00001, prate=0.000001,
                 rmsize=int(60000), window_length=int(1),
                 tau=0.001, init_w=0.003,
                 epsilon=10000, discount=0.99)

    agent.load_weights("../models",step="4")
    agent.eval()

    env.soft_reset()

    for i in range(2001):
        act = agent.select_action(env.struct_obv().float().cuda())

        act = (act + 1) / 2
        act = np.reshape(act, (apNum, 2))
        act[:, 0] = np.round(act[:, 0] * 5)
        act[:, 1] = np.round(act[:, 1] * 5) * 0.002
        print(act)
        reward = env.step(act)

        Rsum["DDPG"].append(env.get_Rsum())
        R5per["DDPG"].append(env.get_R5per())
        rewards["DDPG"].append(reward[1].data)

    print(np.mean(Rsum["DDPG"]),np.mean(R5per["DDPG"]),np.mean(rewards["DDPG"]))

    env.plot_scheduling()


    # env.soft_reset()

    #
    # for i in range(2001):
    #
    #     # actions = np.array([[0,0.01],[0,0.01],[0,0.01],[0,0.01]])
    #     actions = full_reuse(apNum,0.01)
    #     reward = env.step(actions)
    #     Rsum["full-reuse"].append(env.get_Rsum())
    #     R5per["full-reuse"].append(env.get_R5per())
    #     rewards["full-reuse"].append(reward[1].data)
    #
    # env.plot_scheduling()
    #
    # apID = env.apID
    # env.soft_reset()
    # for i in range(2001):
    #
    #     # pos = i % len(apID)
    #     # ap = apID[pos]
    #     # local_id = np.where(apUE[ap] == pos)[0][0]
    #
    #     actions = TDM_single(apNum,i,env.apUE,env.apID,0.01)
    #
    #     # actions = np.array([[0,0.0],[0,0.0],[0,0.0],[0,0.0]])
    #     # actions[ap,0] = local_id
    #     # actions[ap,1] = 0.01
    #
    #     reward = env.step(actions)
    #     Rsum["TDM"].append(env.get_Rsum())
    #     R5per["TDM"].append(env.get_R5per())
    #     rewards["TDM"].append(reward[1].data)

    # env.plot_scheduling()
    #
    # env.soft_reset()





    # plt.figure()
    # plt.plot(Rsum["full-reuse"][100:])
    # plt.plot(Rsum["TDM"][100:])
    # plt.plot(Rsum["DDPG"][100:])
    # plt.legend(["full-reuse","TDM","DDPG"])
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(R5per["full-reuse"][100:])
    # plt.plot(R5per["TDM"][100:])
    # plt.plot(R5per["DDPG"][100:])
    # plt.legend(["full-reuse", "TDM", "DDPG"])
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(rewards["full-reuse"],"r-",alpha = 0.3)
    # plt.plot(rewards["TDM"],"b-", alpha = 0.3)
    # plt.plot(rewards["DDPG"], "g-", alpha = 0.3)
    # plt.plot(signal.savgol_filter(rewards["full-reuse"],51,1),"r.-")
    # plt.plot(signal.savgol_filter(rewards["TDM"],51,1),"b.-")
    # plt.plot(signal.savgol_filter(rewards["DDPG"],51,1), "g.-")
    # plt.legend(["full-reuse", "TDM", "DDPG"])
    # plt.grid()
    # plt.show()






