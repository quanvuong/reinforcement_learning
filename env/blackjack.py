from gym.envs.toy_text import BlackjackEnv

#  temporary fixes for
#  https://github.com/openai/gym/pull/583
class Blackjack(BlackjackEnv):

    def __init__(self, **kwargs):
        self.spec = None
        super().__init__(kwargs)
