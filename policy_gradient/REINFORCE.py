import gym
import sys
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow import nn, train
from collections import namedtuple

tf.logging.set_verbosity(tf.logging.INFO)

sys.path.append('../')
# Import envs here to run __init__ to register the env with gym
import envs

env = gym.make('CliffWalking-v0')

OBSERVATION_SPACE = env.observation_space.n
ACTION_SPACE = env.action_space.n


class PolicyEstimator:
    """
    Policy Function Estimator
    """

    def __init__(self, learning_rate=0.01, scope='policy_estimator'):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], 'state')
            self.action = tf.placeholder(dtype=tf.int32, name='action')
            self.target = tf.placeholder(dtype=tf.float32, name='target')

            # Table look up estimator
            state_one_hot = tf.one_hot(self.state, int(OBSERVATION_SPACE))
            self.output_layer = layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=ACTION_SPACE,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            )

            self.action_probs = tf.squeeze(nn.softmax(self.output_layer))
            self.picked_action_probs = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_probs) * self.target

            self.optimizer = train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator:
    """
    Value Function estimator 
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], name='state')
            self.target = tf.placeholder(tf.float32, name='target')

            # Table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(OBSERVATION_SPACE))
            self.output_layer = layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            )

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=train.get_global_step()
            )

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def reinforce(env, estimator_policy, estimator_value,
              num_episodes=5000, discount=1.0):
    """
    REINFORCE algorithm (Monte Carlo Policy Gradient). Optimize the policy 
     function approximator using policy gradient.
    :param env: OpenAi env
    :param estimator_policy: policy function to be optimized 
    :param estimator_value: Value function estimator, used as baseline 
    :param num_episodes: 
    :param discount: 
    :return: episode lengths and episode rewards
    """

    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    for i_episode in range(num_episodes):
        state = env.reset()

        episode = []

        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Keep track of the transition
            transition = Transition(state, action, reward, next_state, done)
            episode.append(transition)
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            print("\nStep {} @ Episode {}/{} (Reward: {})".format(
                t, i_episode + 1, num_episodes, episode_rewards[i_episode]))
            sys.stdout.flush()

            if done:
                break

            state = next_state

        for timestep, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount ** i * t.reward for i, t in enumerate(episode[timestep:]))
            # Update the value estimator
            estimator_value.update(transition.state, total_return)
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict(transition.state)
            advantage = total_return - baseline_value
            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action)

    return episode_lengths, episode_rewards


def main():
    tf.reset_default_graph()

    global_step = tf.Variable(0, name='global_step', trainable=False)
    policy_estimator = PolicyEstimator()
    value_estimator = ValueEstimator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        episode_lengths, episode_rewards = reinforce(
            env, policy_estimator, value_estimator
        )


if __name__ == '__main__':
    main()
