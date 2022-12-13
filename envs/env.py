import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from scipy import signal
from ddpg import DDPG
from models.baselines import *
from utils.comm_utils import *
import pdb

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


class Env():

    def __init__(self,apNum = 4,ueNum = 24,boarder = 500,UE_move = 1,T = 2000, top_K = 3,random_seed = None):
        self.apNum = apNum
        self.ueNum = ueNum
        self.boarder = boarder
        self.alphaI = 0.05
        self.alphaR = 0.01
        self.gamma = 0.9
        self.lambda_e = 0.8
        self.UE_move = UE_move
        self.T = T
        self.Done = False
        self.top_K = top_K

        if random_seed != None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.reset()



    def reset(self,random_seed = None):



        if random_seed != None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.time = 0
        self.appos = APgen(self.boarder,self.apNum,35)
        self.uepos = UEgen(self.appos,self.boarder,self.ueNum,10)
        self.dists = getDist(self.appos,self.uepos)
        # PathLoss then make the UE-AP association
        self.pathLoss = DSPloss(self.dists)
        self.apID = np.argmax(self.pathLoss,0)

        self.apUE = dict()
        # If any AP doesn't get an associated UE, give the most close UE to that AP
        for i in range(self.apNum):
            self.apUE[i] = np.where(self.apID == i)[0]
            if len(self.apUE[i]) == 0:
                ID = np.argsort(self.pathLoss[i,:])[0]
                self.apUE[i] = np.array([ID])
                pre_AP = self.apID[ID]
                self.apID[ID] = i
                if i < pre_AP:
                    continue
                mask = np.ones(self.apUE[pre_AP].shape, dtype=bool)
                mask[self.apUE[pre_AP] == ID] = False
                self.apUE[pre_AP] = self.apUE[pre_AP][mask]

        self.power = np.zeros(self.dists.shape) + 0.01
        self.accRate = np.zeros((self.ueNum, )) + 1e-2
        self.weight = 1/self.accRate
        self.max_weight = 1
        self.min_weight = 0.1
        self.normalized_weight = state_normalization(self.weight,self.max_weight,self.min_weight,20)
        self.infPower = getInfpower(self.pathLoss, 0.01 , self.apID)
        self.SINR = getSINR(self.pathLoss, self.infPower, -134, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
        self.max_SINR = self.SINRdb.max()
        self.min_SINR = self.SINRdb.min()
        self.normalized_SINR = state_normalization(self.SINRdb,self.max_SINR,self.min_SINR,20)
        self.pf = self.weight * np.log2(1 + self.SINR)
        self.accumulated_rate = self.accRate
        self.reward = 0
        self.Done = False

        #return self.struct_obv()
        return self.struct_obv()


    # def soft_reset(self):
    #
    #     self.time = 0
    #     self.power = np.ones(self.dists.shape) * 0.01
    #     self.accRate = getAccRate(self.pathLoss + self.fading + self.shadowing, self.power, -134, self.apID)
    #     self.weight = 1 / self.accRate
    #     self.max_weight = self.weight.max()
    #     self.min_weight = self.weight.min()
    #     self.normalized_weight = state_normalization(self.weight, self.max_weight, self.min_weight, 20)
    #     self.infPower = getInfpower(self.pathLoss + self.fading + self.shadowing, 0.01, self.apID)
    #     self.SINR = getSINR(self.pathLoss + self.fading + self.shadowing, self.infPower, -134, self.apID)
    #     self.SINRdb = 10 * np.log10(self.SINR)
    #     self.max_SINR = self.SINRdb.max()
    #     self.min_SINR = self.SINRdb.min()
    #     self.normalized_SINR = state_normalization(self.SINRdb, self.max_SINR, self.min_SINR, 20)
    #     self.pf = self.weight * np.log2(1 + self.SINR)
    #     self.accumulated_rate = self.accRate
    #     self.reward = 0
    #     self.actions = torch.tensor(np.zeros((self.apNum * 2,)), dtype=torch.float32)

    def ap_obv(self, apID, obs_num):

        ue_id = self.apUE[apID]
        if len(ue_id) < obs_num:
            ue_weights = np.zeros((obs_num,))
            ue_pf = self.pf[ue_id]
            ue_id = np.argsort(ue_pf)[::-1]
            ue_weights[:len(ue_id)] = self.normalized_weight[ue_id]
            ue_SINR = np.zeros((obs_num,)) - 60
            ue_SINR[:len(ue_id)] = self.normalized_SINR[ue_id]
        else:
            ue_pf = self.pf[ue_id]
            ue_id = np.argsort(ue_pf)[::-1][:obs_num]
            ue_weights = self.normalized_weight[ue_id]
            ue_SINR = self.normalized_SINR[ue_id]

        return [ue_weights,ue_SINR]

    def struct_obv(self, ):
        struct_ob = []
        for i in range(self.apNum):
            struct_ob.append(torch.tensor(self.ap_obv(i,self.top_K), dtype=torch.float32).view(-1))
        struct_ob = torch.stack(struct_ob)
        struct_ob = struct_ob.view(-1)
        return struct_ob

    def step(self, actions):



        self.power *= 0

        ues = []

        # HERE IS A HEURISTIC TRANSFORM

        for i in range(self.apNum):
            ues.append(self.ap_action(i,actions[int(i)]))


        self.fading = rayleigh_fading(self.apNum,self.ueNum)
        self.shadowing = log_norm_shadowing(self.apNum,self.ueNum,7)
        instRate = getAccRate(self.pathLoss + self.fading + self.shadowing,self.power,-134, self.apID)
        mask = np.zeros(instRate.shape)
        for i in range(len(ues)):
            if ues[i] != None:
                mask[ues[i]] = 1
        instRate *= mask
        # pdb.set_trace()
        self.accRate = (1 - self.alphaR) * self.accRate + self.alphaR * instRate
        self.weight = 1/self.accRate
        if self.weight.max() > self.max_weight:
            self.max_weight = self.weight.max()
        if self.weight.min() < self.min_weight:
            self.min_weight = self.weight.min()
        self.normalized_weight = state_normalization(self.weight,self.max_weight,self.min_weight,20)
        self.infPower = (1 - self.alphaI) * self.infPower + \
                        self.alphaI * getInfpower(self.pathLoss + self.fading + self.shadowing, 0.01, self.apID)
        self.SINR = getSINR(self.pathLoss + self.fading + self.shadowing, self.infPower, -134, self.apID)
        self.SINRdb = 10 * np.log10(self.SINR)
        if self.SINRdb.max() > self.max_SINR:
            self.max_SINR = self.SINRdb.max()
        if self.SINRdb.min() < self.min_SINR:
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

        # reward = max(reward, -40)

        self.reward = reward

        self.accumulated_rate += instRate
        self.time += 1
        if self.time >= self.T:
            self.Done = True

        #return self.struct_obv(), torch.tensor(reward, dtype=torch.float32)
        return self.struct_obv(), torch.tensor(reward,dtype=torch.float32), self.Done

    def ap_action(self, apID, ueID):

        ue_id = self.apUE[apID]
        ue_pf = self.pf[ue_id]
        ue_ord = np.argsort(ue_pf)[::-1]
        ue_id = ue_id[ue_ord]


        if ueID >= len(ue_id)  or ueID == self.top_K + 1:
            ue = None
            self.power[apID,:] = 0
        else:
            ue = ue_id[int(ueID)]
            self.power[apID,:] = 0.01

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

        s_rate = (self.accRate)  * 100
        plt.scatter(self.uepos[:,0],self.uepos[:,1],marker="o", s = s_rate, c = self.apID)
        plt.xlim([0,self.boarder])
        plt.ylim([0,self.boarder])
        plt.show()




if __name__ == "__main__":


    apNum = 4
    ueNum = 24
    random_seed = 5

    env = Env(apNum,ueNum,500,random_seed)
    apUE = env.apUE
    Rsum = {"full-reuse":[],"TDM":[],"DDPG":[]}
    R5per = {"full-reuse":[],"TDM":[],"DDPG":[]}
    rewards = {"full-reuse":[],"TDM":[],"DDPG":[]}
    Rscore = {"full-reuse": [], "TDM": [], "DDPG": []}
    obs = env.reset(random_seed)


    for j in range(50):
        env.reset(j)
        for i in range(2001):
            pf = env.pf
            apUE = env.apUE
            act = np.zeros((env.apNum,))

            obs, reward, done = env.step(act)

            Rsum["full-reuse"].append(env.get_Rsum())
            R5per["full-reuse"].append(env.get_R5per())
            rewards["full-reuse"].append(reward.item())
            Rscore["full-reuse"].append(env.get_R5per() * 3 + env.get_Rsum() / env.ueNum)

        # print(np.mean(Rsum["DDPG"]),np.mean(R5per["DDPG"]),np.mean(rewards["DDPG"]))
        print(Rsum["full-reuse"][-1], R5per["full-reuse"][-1], Rscore["full-reuse"][-1])

    # env.plot_scheduling()


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






