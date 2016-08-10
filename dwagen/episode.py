
class Episode:
    """
    Q-learning data container
    """

    def __init__(self, state_v, action_v, action, reward, state_v_prime, is_game_end):
        """
        :param state_v:       chainer.Variable (state before doing action)
        :param action_v:      chainer.Variable
        :param action:        int (representing selected action)
        :param reward:        float (received reward for an action)
        :param state_v_prime: chainer.Variable (state, in which selected action resulted)
        :param is_game_end:   boolean
        """
        self.state_v       = state_v
        self.action_v      = action_v
        self.action        = action
        self.reward        = reward
        self.state_v_prime = state_v_prime
        self.is_game_end   = is_game_end

    def __str__(self):
        return "episode(a={}, r={})".format(self.action, self.reward)

    __repr__ = __str__