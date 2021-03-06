from gym.envs.toy_text import BlackjackEnv
from envs.custom import EnvAbstract

#  spec = None temporary fixes for
#  https://github.com/openai/gym/pull/583
class Blackjack(EnvAbstract, BlackjackEnv):

    def __init__(self, **kwargs):
        self.spec = None
        super().__init__(kwargs)

    def get_possible_actions(self, state):
        pass

    def get_possible_next_states(self, state, action):
        pass

    def get_trans_prob(self, state, action, ns):
        pass

    def get_reward(self, state, action, ns):
        pass

    def get_state_space(self):
        pass