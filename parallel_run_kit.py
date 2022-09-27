import time
import torch
import multiprocessing as mp

from run import run

class ParallelRun():
    def __init__(self, env, agent, max_length, explore_epsilon=0.2, test_mode=False, commu=True):
        self.env = env
        self.agent = agent
        self.max_length = max_length
        self.test_mode = test_mode
        self.explore_epsilon = explore_epsilon
        self.commu = commu
    
    def run(self, batch_size):
        p = mp.get_context('spawn').Pool(batch_size)
        base_seed = time.time()
        seeds = [base_seed + i for i in range(batch_size)]

        batch = p.map(self.randrun, seeds)
        p.close()
        p.join()
        return batch

    def randrun(self, seed):
        torch.random.manual_seed(seed)
        b = run(self.env, self.agent, self.max_length, explore_epsilon=self.explore_epsilon, test_mode = self.test_mode, commu=self.commu)
        return b