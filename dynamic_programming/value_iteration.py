import sys
sys.path.append('../')
import numpy as np
import math
from pprint import pprint
from lib.policy import RandomPolicy
from env.gridworld import GridWorld


def update_rule(policy, env, state, states_values, discount):
    max_sum = - math.inf
    for action in env.get_possible_actions(state):
        for ns in env.get_possible_next_states(state, action):
            trans_prob = env.get_trans_prob(state, action, ns)
            reward = env.get_reward(state, action, ns)
            ns_value = discount * states_values[ns]
            tmp = trans_prob * (reward + ns_value)
            if tmp > max_sum:
                max_sum = tmp
    return max_sum


def value_iteration(policy, env, discount=1.0, stopping_condition=0.0001):
    states_values = np.zeros(env.nS)

    states = env.get_state_space()

    while True:
        loss = 0
        for state in states:
            tmp = states_values[state]
            states_values[state] = update_rule(policy, env, state, states_values, discount)
            loss = max(loss, abs(tmp - states_values[state]))
        if loss < stopping_condition:
            break

    return states_values.reshape(env.shape)


if __name__ == '__main__':
    env = GridWorld()
    policy = RandomPolicy(env)
    pprint(value_iteration(policy, env))
