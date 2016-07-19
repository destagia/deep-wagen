from dwagen import PolicyNetwork
from dwagen import Episode
from chainer import Variable, optimizers
import chainer.functions as F
import numpy as np
import random

import tlogger as tl

class Player:

    INPUT_SIZE = 2304

    def __init__(self):
        self.__episodes = []
        self.__network = PolicyNetwork(Player.INPUT_SIZE)
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)

    def learn_win(self):
        for episode in self.__episodes:
            tl.log("learn", episode.action_v.data[0])
            if episode.selected_action == 0:
                true_action = 0
            else:
                true_action = 1
            tl.log("learn", "True Action Index : " + str(true_action))
            true_action_v = np.asarray([true_action]).astype(np.int32)
            self.__optimizer.update(F.softmax_cross_entropy, episode.action_v, true_action_v)
        self.reset()

    def learn_lose(self):
        for episode in self.__episodes:
            tl.log("learn", episode.action_v.data[0])
            if episode.selected_action == 0:
                true_action = 1
            else:
                true_action = 0
            tl.log("learn", "True Action Index : " + str(true_action))
            true_action_v = np.asarray([true_action]).astype(np.int32)
            self.__optimizer.update(F.softmax_cross_entropy, episode.action_v, true_action_v)
        self.reset()

    def reset(self):
        self.__episodes = []

    def jump(self, environment):
        state_v = Variable(np.asarray(environment).astype(np.float32).reshape(1, Player.INPUT_SIZE))
        action_v = self.__network(state_v)
        action_sum = sum(action_v.data[0])
        target = random.uniform(0, action_sum)

        if target < action.data[0][0]:
            selected_action = 0
        else:
            selected_action = 1

        self.__episodes.append(Episode(state_v, action_v, selected_action))
        tl.log("player", "========== JUMP ==========")
        tl.log("player", target)
        tl.log("player", selected_action)
        tl.log("player", action_v.data[0])
        return selected_action == 0

