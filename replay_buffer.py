"""
This documents uses code from Patrick Emami's DDPG implementation, found here:
https://github.com/pemami4911/deep-rl/tree/master/ddpg

Adapted by Rasmus Loft and Andreas Danebo Jensen
"""



""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import generalFunctions

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.mPosMax = -10000000000.0
        self.mPosMin = 10000000000.0
        self.mSpdMax = -10000000000.0
        self.mSpdMin = 10000000000.0
        self.gAngMax = -10000000000.0
        self.gAngMin = 10000000000.0
        self.gSpdMax = -10000000000.0
        self.gSpdMin = 10000000000.0

    def add(self,experience):
        # self.checkMinMaxState(experience[0])
        #scale states and action before adding to RB.
        oldStateScaled = generalFunctions.scaleState(experience[0])
        actionScaled = generalFunctions.scaleAction(experience[1])
        newStateScaled = generalFunctions.scaleState(experience[4])
        #Reshape
        shapedExp = [np.reshape(oldStateScaled,(4,)),np.reshape(actionScaled,(1,)),experience[2],experience[3],np.reshape(newStateScaled,(4,))]
        if self.count < self.buffer_size: 
            self.buffer.append(shapedExp)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(shapedExp)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
        
    def checkMinMaxState(self,state):
        if (state[0] > self.mPosMax):
            self.mPosMax = state[0]
            print('New mPosMax: ',self.mPosMax)
        if (state[0] < self.mPosMin):
            self.mPosMin = state[0]
            print('New mPosMin: ',self.mPosMin)
        if (state[1] > self.mSpdMax):
            self.mSpdMax = state[1]
            print('New mSpdMax: ',self.mSpdMax)
        if (state[1] < self.mSpdMin):
            self.mSpdMin = state[1]
            print('New mSpdMin: ',self.mSpdMin)
        if (state[2] > self.gAngMax):
            self.gAngMax = state[2]
            print('New gAngMax: ',self.gAngMax)
        if (state[2] < self.gAngMin):
            self.gAngMin = state[2]
            print('New gAngMin: ',self.gAngMin)
        if (state[3] > self.gSpdMax):
            self.gSpdMax = state[3]
            print('New gSpdMax: ',self.gSpdMax)
        if (state[3] < self.gSpdMin):
            self.gSpdMin = state[3]
            print('New gSpdMin: ',self.gSpdMin)

