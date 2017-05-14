class EnvAbstract(object):
    def get_possible_actions(self, state):
        raise NotImplementedError

    def get_possible_next_states(self, state, action):
        raise NotImplementedError

    def get_trans_prob(self, state, action, ns):
        raise NotImplementedError

    def get_reward(self, state, action, ns):
        raise NotImplementedError

    def get_state_space(self):
        raise NotImplementedError