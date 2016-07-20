from dwagen import PolicyNetwork
from dwagen import Episode
from chainer import Variable, optimizers
import chainer.functions as F
import numpy as np
import random

import tlogger as tl

class Player:

    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 36
    FRAME_COUNT = 4

    def __init__(self):
        self.__episodes = []
        self.__network = PolicyNetwork()
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)
        self.__images = []

    def learn_win(self):
        for episode in self.__episodes:
            tl.log("learn", episode.action_v.data[0])
            if episode.selected_action == 0:
                true_action = 0
            else:
                true_action = 1
            tl.log("learn", "True Action Index : " + str(true_action))
            true_action_v = Variable(np.asarray([true_action]).astype(np.int32))
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
            true_action_v = Variable(np.asarray([true_action]).astype(np.int32))
            self.__optimizer.update(F.softmax_cross_entropy, episode.action_v, true_action_v)
        self.reset()

    def reset(self):
        self.__episodes = []
        self.__images = []

    def jump(self, image):

        self.__images.append(image)

        if len(self.__images) < Player.FRAME_COUNT:
            return False

        last_4_frames = self.__images[-Player.FRAME_COUNT:]

        state_as_array = np.asarray(last_4_frames).astype(np.float32)
        state_reshaped = state_as_array.reshape(1, Player.FRAME_COUNT, Player.IMAGE_HEIGHT, Player.IMAGE_WIDTH)
        state_v = Variable(state_reshaped)

        action_v = self.__network(state_v)
        action_sum = sum(action_v.data[0])
        target = random.uniform(0, action_sum)

        if target < action_v.data[0][0]:
            selected_action = 0
        else:
            selected_action = 1

        self.__episodes.append(Episode(state_v, action_v, selected_action))
        tl.log("player", "========== JUMP ==========")
        tl.log("player", target)
        tl.log("player", selected_action)
        tl.log("player", action_v.data[0])
        return selected_action == 0

