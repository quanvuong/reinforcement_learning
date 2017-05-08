import sys
sys.path.append('../')
from policy_evaluation import policy_evaluation, get_random_policy
from env.gridworld import GridWorld

def update_rule()

def policy_iteration(policy, env, states_values, discount=1.0):

    states = env.get_state_space()

    while True:
        for state in states:
            tmp = policy[state]



if __name__ == '__main__':
    env = GridWorld()
    policy = get_random_policy(env)
    states_values = policy_evaluation(policy, env)
    print(policy)
