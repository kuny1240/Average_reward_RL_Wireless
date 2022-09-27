# -*- coding: utf-8 -*-

import torch
from env import Env

env = Env(border=torch.Tensor([40, 40]), 
          enbs = torch.Tensor([[10, 10, 10], 
                               [10, 30, 10], 
                               [10, 10, 30],
                               [10, 30, 30]]), 
          n_ues = 10, 
          noise = 5
          )

env.get_global_obs()
for i in range(4):
    env.get_agent_obs(i)

out_l = []
for _ in range(50):
    l = []
    for _ in range(50):
        r = 0
        for _ in range(400):
            obs, reward = env.step(torch.tensor([0, 0, 0, 0]))
            # obs, reward = env.step(torch.randint(6, (4,)))
            r += 0.02 * (reward - r)
        l.append(r)
    l = torch.tensor(l)
    #torch.save(l, "test_r")
    print(l.mean())
    out_l.append(l.mean())
print(out_l)
torch.save(torch.stack(out_l), "test_r_50")
