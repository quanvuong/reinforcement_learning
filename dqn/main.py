import gym
import itertools
import os
import random
import sys
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow import nn
from tensorflow import summary
from tensorflow import contrib

env = gym.make('Breakout-v4')

print("Action space size: {}".format(env.action_space.n))

observation = env.reset()
OBSERVATION_SHAPE = observation.shape
PROCESSED_SHAPE = [84, 84, 1]
print("Observation space shape: {}".format(observation.shape))

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]


class StateProcessor:
    """
    Processes a raw Atari image. Resize and convert to grayscale.
    """

    def __init__(self):
        # Build tf graph
        with tf.variable_scope('state_process'):
            self.input_state = tf.placeholder(shape=list(OBSERVATION_SHAPE),
                                              dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, PROCESSED_SHAPE[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        
        :param sess: a TF sess object 
        :param state: a OBSERVATION_SHAPE Atari RGB state 
        :return: A processed PROCESSED_SHAPE represented grayscale values
        """
        return sess.run(self.output, {self.input_state: state})


class Estimator:
    """
    Q-value estimator neural network.
    
    This network is used for both the Q-Network and Target Network
    """

    def __init__(self, scope='estimator', summaries_dir=None):
        self.scope = scope
        # Write tensorboard summaries to disk
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir,
                                           "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Build the TF graph
        :return: 
        """
        # Placeholder for input
        # Input are 4 RGB frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4],
                                   dtype=tf.uint8,
                                   name='X')
        # Placeholder for TD Target Value
        self.y_pl = tf.placeholder(shape=[None],
                                   dtype=tf.float32,
                                   name='y')
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None],
                                         dtype=tf.int32,
                                         name='actions')

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = layers.conv2d(X, 32, 8, 4, activation_fn=nn.relu)
        conv2 = layers.conv2d(conv1, 64, 4, 2, activation_fn=nn.relu)
        conv3 = layers.conv2d(conv2, 64, 3, 1, activation_fn=nn.relu)

        # FC layers
        flattened = layers.flatten(conv3)
        fc1 = layers.fully_connected(flattened, 512)
        self.predictions = layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the prediction for the chosen action only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.actions_preds = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate loss
        self.losses = tf.squared_difference(self.y_pl, self.actions_preds)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer params
        # From original papers
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard
        self.summaries = summary.merge([
            summary.scalar('loss', self.loss),
            summary.histogram('loss_hist', self.losses),
            summary.histogram('q_values_hist', self.predictions),
            summary.scalar('max_q_value', tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, state):
        """
        predict action values
        :param sess: TF session 
        :param state: state input of shape [batch_size, 160, 160, 3]
        :return: Tensor of shape [batch_size, len(VALID_ACTIONS))] 
        containing the estimated action values
        """
        return sess.run(self.predictions, {self.X_pl: state})

    def update(self, sess, state, actions, y):
        """
        Update the estimator towards a given target
        :param sess: TF sess
        :param state: State input of shape [batch_size, 4, 160, 160, 3]
        :param actions: Chosen actions of shape [batch_size]
        :param y: Targets of shape [batch_size]
        :return: The calculated loss for the batch
        """

        feed_dict = {self.X_pl: state, self.y_pl: y, self.actions_pl: actions}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def run_test():
    # For Testing....

    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    e = Estimator(scope="test", summaries_dir='test/')
    sp = StateProcessor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Example observation batch
        observation = env.reset()

        observation_p = sp.process(sess, observation)
        observation = np.stack([observation_p] * 4, axis=2)
        observations = np.array([observation] * 2)

        # Test Prediction
        print(e.predict(sess, observations))

        # Test training step
        y = np.array([10.0, 10.0])
        a = np.array([1, 3])
        print(e.update(sess, observations, a, y))


def get_sorted_params(estimator):
    params = [
        t for t in tf.trainable_variables() if t.name.startswith(estimator.scope)
    ]

    return sorted(params, key=lambda v: v.name)


def copy_model_params(sess, estimator1, estimator2):
    """
    :param sess: TF sess object
    :param estimator1: Copy from
    :param estimator2: Copy to
    """
    e1_params = get_sorted_params(estimator1)
    e2_params = get_sorted_params(estimator2)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_eps_greedy_policy(estimator, nA):
    """
    Creates an epsilon greedy policy based on a Q-function approximator and epsilon
    :param estimator: An estimator that returns q values for a given state
    :param nA: number of actions in env
    :return: A function that takes (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA
    """

    def policy_fn(sess, observation, eps):
        actions = np.ones(nA, dtype=float) * eps / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        actions[best_action] += (1.0 - eps)
        return actions

    return policy_fn


def reset_env(sess, env, state_processor):
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    return state


def take_a_step(env, policy, state, eps, sess, state_processor):
    action_probs = policy(sess, state, eps)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    next_state, reward, done, info = env.step(VALID_ACTIONS[action])
    next_state = state_processor.process(sess, next_state)
    # print('next_state before appending: {}'.format(next_state))
    next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
    # print('next_state after appending: {}'.format(next_state))
    return next_state, reward, done, info, action


def DQN(sess,
        env,
        q_estimator,
        target_estimator,
        state_processor,
        logdir,
        num_episodes=10000,
        replay_memory_size=500000,
        replay_memory_init_size=50000,
        target_estimator_update_interval=10000,
        discount=0.99,
        eps_start=1.0,
        eps_end=0.1,
        eps_decay_steps=500000,
        batch_size=32,
        record_video_interval=50):
    """
    Q-learning algorithm for off-policy TD control using neural network function approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy
     
    :param sess: TF sess 
    :param env: openAI env
    :param q_estimator: estimator for q values
    :param target_estimator: estimator for targets
    :param state_processor: a StateProcessor object
    :param num_episodes: numbers of episodes to run for 
    :param logdir: dir to save TF summaries in 
    :param replay_memory_size: size of the replay memory
    :param replay_memory_init_size: number of random experiments to sample when initializing the memory
    :param target_estimator_update_interval: interval to copy params of q_estimator to target_estimator
    :param discount: lambda time discount factor
    :param eps_start:  
    :param eps_end: 
    :param eps_decay_steps: number of steps to decay epsilon over 
    :param batch_size: size of batches to sample from the replay memory
    :param record_video_interval:
    :return: 2 numpy arrays for episode_lengths and episode_rewards
    """

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    replay_memory = []

    episode_lengths = np.zeros(num_episodes)
    episode_rewards = np.zeros(num_episodes)

    # Create dirs for checkpoints and summaries
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'model')
    monitor_path = os.path.join(logdir, 'monitor')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()

    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print('Loading model checkpoint {}...\n'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Get the current time step
    total_time = sess.run(contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(eps_start, eps_end, num=eps_decay_steps)

    # Make the behavior policy
    behavior_policy = make_eps_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS)
    )

    # Init the replay memory with random experience
    print('Init replay memory...')
    state = reset_env(sess, env, state_processor)
    for i in range(replay_memory_init_size):
        eps = epsilons[min(total_time, eps_decay_steps - 1)]
        next_state, reward, done, _, action_taken = take_a_step(env,
                                                                behavior_policy,
                                                                state,
                                                                eps,
                                                                sess,
                                                                state_processor)
        replay_memory.append(Transition(
            state, action_taken, reward, next_state, done
        ))
        print('Added one experience to replay memory. action {}. reward {}. done {}. experience number {}.'.format(
            action_taken, reward, done, i
        ))
        if done:
            state = reset_env(sess, env, state_processor)
        else:
            state = next_state

    # Record videos
    env.monitor.start(monitor_path,
                      resume=True,
                      video_callable=lambda count: count % record_video_interval == 0)

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the env
        state = reset_env(sess, env, state_processor)
        loss = None

        for t in itertools.count():

            # Get epsilon for this step
            eps = epsilons[min(total_time, eps_decay_steps - 1)]

            # Add epsilon to tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=eps,
                                      tag='epsilon')
            q_estimator.summary_writer.add_summary(episode_summary, total_time)

            # Update the target estimator
            if total_time % target_estimator_update_interval == 0:
                copy_model_params(sess, q_estimator, target_estimator)

            # Print out which step we are on, useful for debugging
            print('\nStep {} ({}) @ Episode {}/{}, loss: {}'.format(
                t, total_time, i_episode + 1, num_episodes, loss
            ))
            sys.stdout.flush()

            # Take a step in the env
            next_state, reward, done, _, action_taken = take_a_step(env,
                                                                    behavior_policy,
                                                                    state,
                                                                    eps,
                                                                    sess,
                                                                    state_processor)

            # Pop the first element if replay memory is full
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save new transition to replay memory
            replay_memory.append(Transition(
                state, action_taken, reward, next_state, done
            ))

            # Update recorded statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            # Sample a mini batch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = \
                map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch \
                            + np.invert(done_batch).astype(np.float32) \
                              * discount * np.max(q_values_next, axis=1)

            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_time += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summar()
        episode_summary.value.add(
            simple_value=episode_rewards[i_episode],
            node_name='episode_reward',
            tag_name='episode_reward'
        )
        episode_summary.value.add(
            simple_value=episode_lengths[i_episode],
            node_name='episode_reward',
            tag_name='episode_reward'
        )
        q_estimator.summary_writer.add_summary(episode_summary, total_time)
        q_estimator.summary_writer.flush()

        yield total_time, episode_rewards[-1]

    return episode_lengths, episode_rewards


def main():
    tf.reset_default_graph()

    logdir = os.path.abspath('./experiments/{}'.format(env.spec.id))

    # Create a global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope='q', summaries_dir=logdir)
    target_estimator = Estimator(scope='target')

    state_processor = StateProcessor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for timestep, reward in DQN(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    logdir=logdir):
            print('\nEpisode {} reward: {}'.format(
                timestep, reward
            ))


if __name__ == '__main__':
    main()
