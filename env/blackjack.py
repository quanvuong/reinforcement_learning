from gym.envs.toy_text import BlackjackEnv
from env.custom import EnvAbstract

#  spec = None temporary fixes for
#  https://github.com/openai/gym/pull/583
class Blackjack(EnvAbstract, BlackjackEnv):

    def __init__(self, **kwargs):
        self.spec = None
        super().__init__(kwargs)
