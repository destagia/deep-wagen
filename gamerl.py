#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import matplotlib.pyplot as plt


class Inferrer(chainer.Chain):

    def __init__(
        self,
        shape=(64, 64),
        ksize=3,
        channels=[3, 64, 128, 256, 512],
        strides=[1,  2,   2,   2],
        use_fc=None,
        use_bn=None,
        func=F.leaky_relu,
    ):
        super(Inferrer, self).__init__()
        assert len(channels)-1 == len(strides)
        self.shape = shape if hasattr(shape, "__len__") else (shape, shape)
        self.channels = channels
        self.strides = strides
        self.use_fc = use_fc
        self.use_bn = use_bn
        self.func = func
        pad = ksize//2
        self.add_link("conv", chainer.ChainList())
        for i in range(1, len(self.channels)):
            self.conv.add_link(L.Convolution2D(
                self.channels[i-1],  self.channels[i],
                ksize=ksize, stride=self.strides[i-1], pad=pad,
                wscale=0.02*np.sqrt(ksize*ksize *self.channels[i-1]),
            ))
        if self.use_fc is not None:
            self.add_link("fc", L.Linear(self.channels[-1]*(np.prod(self.shape)//np.prod(self.strides)**2), self.use_fc))
        if self.use_bn:
            self.add_link("bn", chainer.ChainList())
            for i in range(len(self.conv)):
                self.bn.add_link(L.BatchNormalization(self.conv[i].W.data.shape[0], decay=0.9))

    def __call__(self, x, train=False):
        h = x
        for i in range(len(self.conv)):
            k = self.conv[i](h)
            if self.use_bn:
                k =self.bn[i](k, test=not train)
            if i < len(self.conv)-1:
                k = self.func(k)
            h = k
            # print("inf h[{}]".format(i),h.data.shape)
        if self.use_fc:
            h = self.func(h)
            h = self.fc(h)
        return h


class Game:

    """docstring for Game"""

    def __init__(
        self,
        shape=(16, 8),  # height x width of play window
        difficulty=0.3,  # ratio of block
        pos=None,       # initial player position
    ):
        self.shape = shape
        self.difficulty = difficulty
        self.pos = pos
        self.n_action = 5
        self.initialize()

    def initialize(self):
        if self.pos is None:
            self.pos = (self.shape[0]//2, self.shape[1]//2)
        self.blocks = np.zeros(self.shape, dtype=np.float32)
        self.reward = 0

    def get_state(self):
        player = np.zeros(self.shape, dtype=np.float32)
        player[self.pos] = 1
        state = np.concatenate([[self.blocks], [player]], axis=0)
        return state.reshape((1, )+state.shape)

    def get_state_visible(self):
        state =self.get_state()[0]
        state = state[0] + state[1]*2
        vstate = np.full(self.shape, " ", dtype=str)
        vstate[state == 1] = "+"
        vstate[state == 2] = "A"
        vstate[state == 3] = "X"
        vstate = "\n".join(["".join(v_)  for v_ in vstate])
        return vstate

    def get_reward(self):
        return self.reward

    def transit(self, action):
        # wait, up, down, left, right
        assert action in [0, 1, 2, 3, 4]
        # move player
        mov = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), ][action]
        self.pos = ((self.pos[0]+mov[0]) %self.shape[0], (self.pos[1]+mov[1]) %self.shape[1])
        # make new blocks
        self.blocks[1:] =self.blocks[:-1]
        self.blocks[0] = np.random.random(self.shape[1]) <self.difficulty
        # make a reward
        penalty = int(self.blocks[self.pos] == 1)
        self.reward = -penalty  # +self.difficulty


class DeepRL(chainer.Chain):

    """reinforcement learning"""

    def __init__(
        self,
        game,
        discount_factor=0.5
    ):
        super(DeepRL, self).__init__()
        self.game = game
        self.q_a_given_s = Inferrer(
            shape=self.game.shape,
            ksize=3,
            channels=[2, 32, 64, 128],
            strides=[1,  2,  2],
            use_fc=self.game.n_action,
            use_bn=None,
            func=F.leaky_relu,
        )
        self.discount_factor = discount_factor
        self.shape =self.game.get_state().shape
        self.opt = chainer.optimizers.Adam()
        self.opt.setup(self.q_a_given_s)

    def policy_greedy(self, q):
        return np.argmax(q)

    def policy_egreedy(self, q, e=0.1):
        if np.random.random() < e:
            return np.random.choice(len(q))
        else:
            return np.argmax(q)

    def policy_e_softmax(self, q):
        t = 0.1
        expq = np.exp(q/t)
        return np.random.choice(len(expq), p=expq/np.sum(expq))

    def policy_random(self, q):
        return np.random.choice(len(q))

    def get_policy(self, policy=None):
        if policy == "greedy":
            policy =self.policy_greedy
        elif policy == "egreedy":
            policy =self.policy_egreedy
        elif policy == "esoftmax":
            policy =self.policy_e_softmax
        else:
            policy =self.policy_e_softmax
        return policy

    def train(self, n_step=10000, policy=None):
        policy =self.get_policy(policy)
        state      = np.empty(shape=(n_step+1,) +self.shape, dtype=np.float32)
        action     = np.empty(shape=(n_step+1,),        dtype=np.int32)
        value_max  = np.empty(shape=(n_step+1,),        dtype=np.float32)
        value_act  = [None for i in range(n_step+1)]  # list of chainer.Variable
        reward     = np.empty(shape=(n_step+1,),        dtype=np.float32)
        for t in range(n_step+1):
            reward[t]     = game.get_reward()
            state[t]      = self.game.get_state()
            value_all     = F.reshape(self.q_a_given_s(state[t]),(-1,))
            action[t]     = policy(value_all.data)
            value_act[t]  = F.get_item(value_all, slice(action[t], action[t]+1))
            value_max[t]  = np.max(value_act[t].data)
            game.transit(action[t])

        value_predicted = F.get_item(F.concat(value_act, axis=0), (slice(0, n_step), ))
        value_actual    = (reward[1:] +self.discount_factor*value_max[1:])
        self.opt.update(F.mean_squared_error, value_predicted, value_actual)
        return reward.sum()

    def play(self, n_step=100, policy=None):
        import time
        policy =self.get_policy(policy)
        reward = 0
        for itr in range(n_step):
            state  = self.game.get_state()
            value  = F.reshape(self.q_a_given_s(state),(-1,))
            action = policy(value.data)
            game.transit(action)
            reward += game.get_reward()
            print(
                "--------------------------------------------",
                game.get_state_visible(),
                "reward: {}".format(reward),
                "step: {}".format(itr+1),
                sep="\n"
            )
            time.sleep(0.1)
        return reward

if __name__ == '__main__':
    # setting
    learning_itrmax = 1000000
    learning_step   = 100
    play_interval   = 50
    play_step       = 100
    plot_interval   = 10
    difficulty      = 0.3
    discount_factor = 0.5
    reward_plot_decay = 0.5
    policy_learning = "esoftmax"
    policy_testplay = "esoftmax"
    # plot reward
    historical_reward = []
    fig, ax = plt.subplots(1, 1)
    lines, = ax.plot([0], [0])
    plt.xlabel("itration")
    plt.ylabel("reward (moving average with decay of {})".format(reward_plot_decay))
    # initialize game and AI
    game = Game(difficulty=difficulty)
    rl = DeepRL(game, discount_factor=discount_factor)
    # main loop
    for itr in range(learning_itrmax):
        # plot reward
        if itr % plot_interval == 0 and itr>0:
            lines.set_data(range(len(historical_reward)), historical_reward)
            ax.set_xlim((0, len(historical_reward)))
            ax.set_ylim((np.min(historical_reward), 0))
            plt.waitforbuttonpress(timeout=.01)
        # test play
        if itr % play_interval == 0:
            rl.play(n_step=play_step, policy=policy_testplay)
        # learning
        reward = rl.train(n_step=learning_step, policy=policy_learning)
        average_reward=reward/learning_step
        print("step", itr, "average reward", average_reward)
        if itr>0:
            average_reward=historical_reward[-1]*reward_plot_decay+average_reward*(1-reward_plot_decay)
        historical_reward.append(average_reward)
