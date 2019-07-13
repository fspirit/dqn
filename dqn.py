import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import sys

if "../" not in sys.path:
  sys.path.append("../")

from tf_board_logger import TFBoardLogger
from epsilon_decay_schedule import EpsilonDecaySchedule
from common import *
from replay_memory import ReplayMemory
from make_policy import make_epsilon_greedy_policy
from tf_lib import TensorFlow

class EpisodeStatsStorage(object):

    def __init__(self, num_episodes):
        self.episode_lengths = np.zeros(num_episodes)
        self.episode_rewards = np.zeros(num_episodes)

    def update_episode_stats(self, index, reward, length):
        self.episode_rewards[index] += reward
        self.episode_lengths[index] = length

    def get_episode_stats(self, index):
        return self.episode_rewards[index], self.episode_lengths[index]


class DeepQLearner(object):

    def __init__(self, lib, env, working_dir, record_video_every=50):
        self.lib = lib

        # FIXME: Dep on TFBoard, how to switch to pyTorch?
        self.tf_board_logger = TFBoardLogger("q", working_dir)

        self.total_steps = lib.get_number_of_steps_done()

        self.epsilon_decay_schedule = EpsilonDecaySchedule(1.0, 0.1, 500000)

        self.q_estimator = lib.get_q_estimator(self.tf_board_logger)
        self.target_estimator = lib.get_target_estimator()

        self.state_processor = lib.get_state_processor()

        self.policy = make_epsilon_greedy_policy(self.q_estimator, len(VALID_ACTIONS))

        self.replay_memory = ReplayMemory()
        self.replay_memory.init_replay_memory(env, self.policy, self.state_processor, self.epsilon_decay_schedule, self.total_steps)

        self.env_wrapper = Monitor(env,
                      directory=os.path.join(working_dir, "monitor"),
                      resume=True,
                      video_callable=lambda count: count % record_video_every == 0)

    def reset_env(self):
        state = self.env_wrapper.reset()
        state = self.state_processor.process(state)
        state = np.stack([state] * 4, axis=2)

        return state

    def step(self, state, action):
        next_state, reward, done, _ = self.env_wrapper.step(VALID_ACTIONS[action])
        next_state = self.state_processor.process(next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

        return next_state, reward, done

    def log_step(self, timestep, episode, total_episodes_count, loss):
        print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
            timestep, self.total_steps, episode + 1, total_episodes_count, loss), end="")
        sys.stdout.flush()

    def run(self,
            episodes_to_run,
            update_target_estimator_every=10000,
            discount_factor=0.99,
            batch_size=32):

        stats_storage = EpisodeStatsStorage(episodes_to_run)

        # TODO: Shorten this function as much as possible
        for episode in range(episodes_to_run):

            # Save the current state of models
            self.lib.save()

            # Reset the environment
            state = self.reset_env()
            loss = None

            # One step in the environment
            for timestep in itertools.count():

                # Epsilon for this time step
                epsilon = self.epsilon_decay_schedule.next_epsilon(self.total_steps)

                # Add epsilon to Tensorboard
                # TODO: put together all ther logging into some structure - single interface with appenders
                # TODO: and add lib specific loggers inside lib.addLoggers(log)
                self.tf_board_logger.log_epsilon(epsilon, self.total_steps)

                # Maybe update the target estimator
                if self.total_steps % update_target_estimator_every == 0:
                    self.target_estimator.copy_parameters_from(self.q_estimator)

                # Print out which step we're on, useful for debugging.
                self.log_step(timestep, episode, episodes_to_run, loss)

                action = self.policy(state, epsilon)
                next_state, reward, done = self.step(state, action)

                # Save transition to replay memory
                self.replay_memory.append(Transition(state, action, reward, next_state, done))

                # Update statistics
                stats_storage.update_episode_stats(episode, reward, timestep)

                # Sample a minibatch from the replay memory
                samples = self.replay_memory.sample(batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                q_values_next = self.q_estimator.predict(next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = self.target_estimator.predict(next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = self.q_estimator.update(states_batch, action_batch, targets_batch)

                if done:
                    break

                state = next_state
                self.total_steps += 1

            reward, length = stats_storage.get_episode_stats(episode)
            self.tf_board_logger.log_episode_stats(reward, length, self.total_steps)

            yield self.total_steps, reward

        self.env_wrapper.monitor.close()
        return stats_storage


if __name__ == "__main__":
    env = gym.envs.make("Breakout-v0")
    base_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    tf_lib = TensorFlow(base_dir)
    with tf_lib:
        dqn = DeepQLearner(tf_lib, env, base_dir)

        for t, reward in dqn.run(10000):
            print("\nEpisode Reward: {}".format(reward))
