class RandomPolicy(object):

    def __init__(self, env):
        random_policy = {}
        for state in env.get_state_space():
            actions = env.get_possible_actions(state)
            random_policy[state] = {}
            for action in actions:
                random_policy[state][action] = 1.0 / len(actions)  # 1/4 (equal likelihood among actions)
        self.policy = random_policy

    def __getitem__(self, item):
        return self.policy[item]

    def __setitem__(self, key, value):
        self.policy[key] = value
