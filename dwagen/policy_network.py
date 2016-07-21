from chainer import Chain

import chainer.functions as F
import chainer.links     as L
import numpy             as np
import tlogger           as tl


class PolicyNetwork(Chain):
    def __init__(self):
        super(PolicyNetwork, self).__init__(
            conv1=F.Convolution2D(4,  32, ksize=8),
            conv2=F.Convolution2D(32, 64, ksize=4),
            conv3=F.Convolution2D(64, 64, ksize=3),
            l1=F.Linear(79872, 512),
            l2=F.Linear(512, 2))

    def __call__(self, state):
        h1 = F.relu(self.conv1(state))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.l1(h3))
        h5 = F.relu(self.l2(h4))
        tl.log("player", h5.data)
        return F.softmax(h5)
