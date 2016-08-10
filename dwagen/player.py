from dwagen import PolicyNetwork
from dwagen import Episode
from chainer import Variable, optimizers
import chainer.serializers as S
import chainer.functions as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os.path
from collections import deque

import tlogger as tl

class Player:

    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 36
    FRAME_COUNT = 4

    BATCH = 32
    GAMMA = 0.99

    OBSERVE_FRAME = 3200
    REPLAY_MEMORY = 50000

    def __init__(self):
        fig, self.ax = plt.subplots(1, 1)
        self.lines, = self.ax.plot([0], [0])
        plt.xlabel('episode')
        plt.ylabel('reward')
        self.__reward_history = []
        self.__images = deque()
        self.__episodes = deque()
        self.__network = PolicyNetwork()
        self.__optimizer = optimizers.Adam()
        self.__timestamp = 0
        self.load()
        self.__optimizer.setup(self.__network)
        self.reset()

    def save(self):
        S.save_hdf5('network.model', self.__network)
        S.save_hdf5('optimizer.model', self.__optimizer)
        tl.log("save model and optimizer")

    def load(self):
        if os.path.isfile('network.model'):
            tl.log("network model was found! load!")
            S.load_hdf5('network.model', self.__network)
        else:
            tl.log("no model file!")
        # if os.path.isfile('optimizer.model'):
        #     tl.log("optimizer model was found! load!")
        #     S.load_hdf5('optimizer.model', self.__optimizer)

    def reset(self):
        self.__game_start_index = 0
        self.__prev_state_v = None
        self.__prev_action_v = None
        self.__prev_selected_action = None
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

            # create random input data
            inputs[i:i+1] = state_v.data
            targets[i] = self.__network(state_v).data
            Q_sa = self.__network(state_v_prime)

            tl.log("learn", "reward: " + str(reward) + ", action: " + str(action))
            tl.log("learn", "is game end?" + str(is_game_end))

            if is_game_end:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + Player.GAMMA * np.max(Q_sa.data)

        tl.log("learn", targets)
        x = self.__network(inputs)
        self.__optimizer.update(F.MeanSquaredError(), x, targets.astype(np.float32))
        self.save()

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
        elif len(self.__images) > Player.FRAME_COUNT:
            self.__images.popleft()

        # Take last 4 frame
        last_4_frames = list(self.__images)

        state_as_array = np.asarray(last_4_frames).astype(np.float32)
        state_reshaped = state_as_array.reshape(1, Player.FRAME_COUNT, Player.IMAGE_HEIGHT, Player.IMAGE_WIDTH)
        state_v = Variable(state_reshaped)

        action_v = self.__network(state_v)
        if action_v.data[0, 0] == action_v.data[0, 1]:
            selected_action = random.randint(0, 1)
        else:
            selected_action = np.argmax(action_v.data)
        # selected_action = 0 if random.uniform(0, 1) < 0.2 else 1

        if self.__ready_to_store_episode:
            epi = Episode(self.__prev_state_v,
                          self.__prev_action_v,
                          self.__prev_selected_action,
                          reward,
                          state_v,
                          is_game_end)
            tl.log("player", "Q Values= {}, Action= {}, Reward= {}".format(self.__prev_action_v.data[0],
                                                                           self.__prev_selected_action,
                                                                           reward))
            self.__episodes.append(epi)
            if len(self.__episodes) > Player.REPLAY_MEMORY:
                self.__episodes.popleft()

        if is_game_end:
            reward_sum = sum(map(lambda epi: epi.reward, self.__episodes))
            self.__reward_history.append(reward_sum)
            rewards = self.__reward_history[self.__game_start_index:len(self.__reward_history)]
            self.__game_start_index = len(self.__reward_history)
            self.ax.set_xlim((0, len(rewards)))
            self.ax.set_ylim((np.min(rewards), np.max(rewards)))
            self.lines.set_data(range(len(rewards)), rewards)
            # plt.show()

        if self.__timestamp > Player.OBSERVE_FRAME:
            self.learn()

        self.__timestamp += 1

        self.__prev_state_v = state_v
        self.__prev_action_v = action_v
        self.__prev_selected_action = selected_action
        self.__ready_to_store_episode = True

        return selected_action == 0

class LossFunction:

    def __call__(self, x, t):
        tl.log("loss", x.data)
        tl.log("loss", t.data)
        loss = F.sum(F.log(F.softmax(t - x))) / len(x.data)
        tl.log("loss", loss.data)
        return loss
