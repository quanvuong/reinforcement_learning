import sys
import numpy as np
import math
from policy_evaluation import policy_evaluation
from pprint import pprint
sys.path.append('../')
from env.gridworld import GridWorld
from lib.policy import RandomPolicy


def update_rule(policy, env, state, states_values, discount):
    argmax_action = None
    max_sum = - math.inf
    for action in env.get_possible_actions(state):
        for ns in env.get_possible_next_states(state, action):
            trans_prob = env.get_trans_prob(state, action, ns)
            reward = env.get_reward(state, action, ns)
            ns_value = discount * states_values[ns]
            tmp = trans_prob * (reward + ns_value)
            if tmp > max_sum:
                max_sum = tmp
                argmax_action = action
    return argmax_action


def policy_iteration(policy, env, discount=1.0):
    states = env.get_state_space()

    while True:
        policy_stable = True
        states_values = policy_evaluation(policy, env)
        states_values = states_values.flatten()
        for state in states:
            tmp = policy[state]
            argmax_action = update_rule(policy, env, state, states_values, discount)
            for action in policy[state]:
                if action == argmax_action:
                    policy[state][action] = 1.0  # Max prob
                else:
                    policy[state][action] = 0
            if tmp != policy[state]:
                policy_stable = False
        if policy_stable:
            return policy


if __name__ == '__main__':
    env = GridWorld()
    policy = RandomPolicy(env)
    pprint(policy_iteration(policy, env).__dict__)
