import gym
from gym.envs.registration import register

register(
    id='CliffWalking-v0',
    entry_point='envs.cliffwalking:CliffWalking',
)