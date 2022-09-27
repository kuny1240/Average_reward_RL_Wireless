# -*- Coding: utf-8 -*-

import torch
import random

class Memory(object):
    def __init__(self, buffer_limit=1000):
        super().__init__()
        self.buffer = []
        self.length = 0
        self.buffer_limit = buffer_limit
    
    def remember(self, data):
        if self.buffer == []:
            self.buffer = data
        else:
            self.buffer = torch.cat(self.buffer, data, dim=0)
        self.length += data.shape[0]
        self.forget()

    # random sample
    def sample(self, batch_size, cuda_flag = False):
        inds = random.sample(range(self.length), batch_size)
        batch = self.buffer[inds].clone()
        if cuda_flag:
            for key in batch.keys():
                batch[key] = batch[key].cuda()
        return batch

    def forget(self):
        if self.length > self.buffer_limit:
            self.buffer = self.buffer[-self.buffer_limit:]
            self.length = self.buffer_limit
            
            
class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=64):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(self.seq_length + 1, len(self.memory)), batch_size)
        begin = [x - self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            # correct for sampling near beginning
            final = self.memory[max(start + 1, 0):end + 1]

            # correct for sampling across episodes
            for i in range(len(final) - 2, -1, -1):
                if final[i][3] is None:
                    final = final[i + 1:]
                    break

            # pad beginning to account for corrections
            while (len(final) < self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]))] + final

            samp += final

        # returns flattened version
        return samp, None, None

    def __len__(self):
        return len(self.memory)
