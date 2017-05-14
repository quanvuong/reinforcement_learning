import sys
sys.path.append('../')
import numpy as np
from pprint import pprint
from envs.blackjack import Blackjack
from lib.policy import RandomPolicy
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg')


def policy_evaluation(policy, env, num_episodes, discount=1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):

        if i_episode % 1000 == 0:
            print('\nEpisode {}/{}.'.format(
                i_episode, num_episodes
            ), end="")
            sys.stdout.flush()

        episode = []

        state = env.reset()

        for t in range(100):
            action = policy(state)
            ns, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = ns

        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state)
            G = sum([x[2] * (discount**i) for i, x in enumerate(
                episode[first_occurence_idx:]
            )])

            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    env = Blackjack()

    V_10k = policy_evaluation(sample_policy, env, num_episodes=10000)
    pprint(V_10k)

    V_500k = policy_evaluation(sample_policy, env, num_episodes=500000)
    pprint(V_500k)
