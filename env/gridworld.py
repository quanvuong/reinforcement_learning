from gym.envs.toy_text import discrete
import numpy as np
from enum import Enum
import sys
from env.custom import EnvAbstract

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridWorld(EnvAbstract, discrete.DiscreteEnv):
    '''Sutton Example 4.1'''

    metadata = {'render.modes': ['human']}

    def __init__(self, shape=[4, 4]):
        self.shape = shape
        self.spec = None
        self.possible_next_states = {}

        nS = np.prod(shape)
        nA = len(Action)

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        self.grid = np.arange(nS).reshape(shape)
        iter = np.nditer(self.grid, flags=['multi_index'])

        def is_terminal(state):
            return state == 0 or state == (nS - 1)

        while not iter.finished:
            s = iter.iterindex
            y, x = iter.multi_index

            P[s] = {a: [] for a in Action}

            reward = 0.0 if is_terminal(s) else -1.0

            if is_terminal(s):
                for a in Action:
                    #  prob, next state, reward, done
                    P[s][a] = [(1.0, s, reward, True)]
                    self.possible_next_states[s] = {}
                    for action in Action:
                        self.possible_next_states[s][action] = [s]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_left = s if x == 0 else s - 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                P[s][Action.UP] = [(1.0, ns_up, reward, is_terminal(ns_up))]
                P[s][Action.RIGHT] = [(1.0, ns_right, reward, is_terminal(ns_right))]
                P[s][Action.LEFT] = [(1.0, ns_left, reward, is_terminal(ns_left))]
                P[s][Action.DOWN] = [(1.0, ns_down, reward, is_terminal(ns_down))]
                self.possible_next_states[s] = {}
                self.possible_next_states[s][Action.UP] = [ns_up]
                self.possible_next_states[s][Action.RIGHT] = [ns_right]
                self.possible_next_states[s][Action.LEFT] = [ns_left]
                self.possible_next_states[s][Action.DOWN] = [ns_down]

            iter.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        self.P = P

        super().__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = sys.stdout

        iter = np.nditer(self.grid, flags=['multi_index'])
        while not iter.finished:
            s = iter.iterindex
            y, x = iter.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            iter.iternext()

    def get_possible_actions(self, state):
        return Action

    def get_possible_next_states(self, state, action):
        return self.possible_next_states[state][action]

    def get_trans_prob(self, state, action, ns):
        possible_trans = self.P[state][action]
        for trans in possible_trans:
            prob, next_state, _, _ = trans
            if ns == next_state:
                return prob

    def get_reward(self, state, action, ns):
        possible_trans = self.P[state][action]
        for trans in possible_trans:
            _, next_state, reward, _ = trans
            if ns == next_state:
                return reward

    @staticmethod
    def get_action_enum():
        return Action

    def get_state_space(self):
        return self.P.keys()