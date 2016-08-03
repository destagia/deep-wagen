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

    BATCH = 32
    GAMMA = 0.99

    def __init__(self):
        self.__network = PolicyNetwork()
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)
        self.reset()

    def reset(self):
        self.__episodes = []
        self.__images = []
        self.__prev_state_v = None
        self.__prev_action_v = None
        self.__prev_is_game_end = None
        self.__ready_to_store_episode = False

    def learn(self):
        minibatch = random.sample(self.__episodes, Player.BATCH)

        inputs = np.zeros((Player.BATCH, Player.FRAME_COUNT, Player.IMAGE_HEIGHT, Player.IMAGE_WIDTH)).astype(np.float32)
        targets = np.zeros((inputs.shape[0], 2))

        for i in range(0, len(minibatch)):
            data = minibatch[i]
            state_v = data.state_v
            action = data.action
            action_v = data.action_v
            reward = data.reward
            state_v_prime = data.state_v_prime
            is_game_end = data.is_game_end

            inputs[i:i+1] = state_v.data
            targets[i] = self.__network(state_v).data
            Q_sa = self.__network(state_v_prime)

            tl.log("minibatch", "reward: " + str(reward) + ", action: " + str(action))

            if is_game_end:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + Player.GAMMA * np.max(Q_sa.data)

        x = self.__network(inputs)
        self.__optimizer.update(LossFunction(), x, targets)

    def jump(self, image, reward, is_game_end):
        """
        Reward setting:
            -1  : when car crashed
            0.1 : when car jumped over a hole!
            1   : when car reached goal (300 meters point from start)

        :param image:       Compressed game screen shot (64 * 36)
        :param reward:      Reward PREVIOUS action caused
        :param is_game_end: Flag for checking whether or not game reached end
        :return:            Action against given frame
        """

        self.__images.append(image)

        if len(self.__images) < Player.FRAME_COUNT:
            return False

        # Take last 4 frame
        last_4_frames = self.__images[-Player.FRAME_COUNT:]

        state_as_array = np.asarray(last_4_frames).astype(np.float32)
        state_reshaped = state_as_array.reshape(1, Player.FRAME_COUNT, Player.IMAGE_HEIGHT, Player.IMAGE_WIDTH)
        state_v = Variable(state_reshaped)

        action_v = self.__network(state_v)
        selected_action = np.argmax(action_v.data)

        if self.__ready_to_store_episode:
            self.__episodes.append(Episode(self.__prev_state_v,
                                           self.__prev_action_v,
                                           selected_action,
                                           reward,
                                           state_v,
                                           self.__prev_is_game_end))

        tl.log("player", "========== JUMP ==========")
        tl.log("player", action_v.data[0])
        tl.log("player", "selected action : " + str(selected_action))

        self.__prev_state_v = state_v
        self.__prev_action_v = action_v
        self.__prev_is_game_end = is_game_end
        self.__ready_to_store_episode = True

        return selected_action == 0

class LossFunction:

    def __call__(self, x, t):
        loss = F.sum(F.log(F.softmax(t - x))) / len(x.data)
        tl.log("loss", loss.data)
        return loss
