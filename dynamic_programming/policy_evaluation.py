import numpy as np
import sys
sys.path.append('../')
from env.gridworld import GridWorld


def update_rule(policy, env, state, states_values, discount):
    value = 0
    for action in env.get_possible_actions(state):
        action_prob = policy[state][action]

        next_states = env.get_possible_next_states(state, action)

        for ns in next_states:
            trans_prob = env.get_trans_prob(state, action, ns)
            reward = env.get_reward(state, action, ns)
            ns_value = states_values[ns]
            value += action_prob * trans_prob * (reward + discount * ns_value)
    return value


def policy_evaluation(policy, env, discount=1.0, stopping_condition=0.0001):
    # Init state value function
    states_values = np.zeros(env.nS)

    # Init loss

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


def get_random_policy(env):
    random_policy = {}
    for state in env.get_state_space():
        actions = env.get_possible_actions(state)
        random_policy[state] = {}
        for action in actions:
            random_policy[state][action] = 0.25  # 1/4 (equal likelihood among actions)
    return random_policy

if __name__ == '__main__':
    env = GridWorld()
    policy = get_random_policy(env)
    print(policy_evaluation(policy, env))
