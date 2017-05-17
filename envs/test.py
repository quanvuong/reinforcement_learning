
env = gym.make('CliffWalking-v0')

for i_episode in range(1):
    observation = env.reset()
    for t in itertools.count():
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} timesteps'.format(
                t+1
            ))
            break
