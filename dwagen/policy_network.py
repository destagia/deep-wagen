from chainer import Chain

import chainer.functions as F
import chainer.links     as L
import numpy             as np


class PolicyNetwork(Chain):
    def __init__(self, n_input):
        super(PolicyNetwork, self).__init__(
            l1=F.Linear(n_input, 1000),
            l2=F.Linear(1000, 1000),
            l3=F.Linear(1000, 2))

    def __call__(self, state):
        h1 = F.relu(self.l1(state))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return F.softmax(h3)

