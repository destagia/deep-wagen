from dwagen import PolicyNetwork
from dwagen import Episode
from chainer import Variable, optimizers
import chainer.functions as F
import numpy as np

class Player:

    INPUT_SIZE = 2304

    def __init__(self):
        self.__episodes = []
        self.__network = PolicyNetwork(Player.INPUT_SIZE)
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)

    def learn_win(self):
        for episode in self.__episodes:
            if episode.action_v.data[0][0] > episode.action_v.data[0][1]:
                true_action = 0
            else:
                true_action = 1
            true_action_v = np.asarray([true_action]).astype(np.int32)
            self.__optimizer.update(F.softmax_cross_entropy, episode.action_v, true_action_v)
        self.reset()

    def learn_lose(self):
        for episode in self.__episodes:
            if episode.action_v.data[0][0] > episode.action_v.data[0][1]:
                true_action = 1
            else:
                true_action = 0
            true_action_v = np.asarray([true_action]).astype(np.int32)
            self.__optimizer.update(F.softmax_cross_entropy, episode.action_v, true_action_v)
        self.reset()

    def reset(self):
        self.__episodes = []

    def jump(self, environment):
        state_v = Variable(np.asarray(environment).astype(np.float32).reshape(1, Player.INPUT_SIZE))
        action_v = self.__network(state_v)
        self.__episodes.append(Episode(state_v, action_v))
        print("========== JUMP ==========")
        print(action_v.data[0])
        return action_v.data[0][0] > action_v.data[0][1]

