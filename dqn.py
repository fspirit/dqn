import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os

from console_logger import ConsoleLogger
from log import Log

from tf_board_logger import TFBoardLogger
from epsilon_decay_schedule import EpsilonDecaySchedule
from common import *
from replay_memory import ReplayMemory
from make_policy import make_epsilon_greedy_policy
from tf_lib import TensorFlow

class DeepQLearner(object):

    def __init__(self, lib, env, working_dir, record_video_every=50):
        self.lib = lib

        self.total_steps = lib.get_number_of_steps_done()

        self.epsilon_decay_schedule = EpsilonDecaySchedule(1.0, 0.1, 1e6)

        self.q_estimator = lib.get_q_estimator()
        self.target_estimator = lib.get_target_estimator()
        self.state_processor = lib.get_state_processor()

        self.policy = make_epsilon_greedy_policy(self.q_estimator, len(VALID_ACTIONS))

        self.replay_memory = ReplayMemory()
        self.replay_memory.init_replay_memory(env,
                                              self.policy,
                                              self.state_processor,
                                              lambda: self.epsilon_decay_schedule.next_epsilon(self.total_steps))

        self.env_wrapper = Monitor(env,
                      directory=os.path.join(working_dir, "monitor"),
                      resume=True,
                      video_callable=lambda count: count % record_video_every == 0)

        self._set_up_logging(working_dir)

    def _set_up_logging(self, base_dir):
        self.log = Log()
        self.log.add_logger(ConsoleLogger())
        self.log.add_logger(TFBoardLogger(base_dir))

    def reset_env(self):
        state = self.env_wrapper.reset()
        state = self.state_processor.process(state)
        state = np.stack([state] * 4, axis=2)

        return state

    def step(self, state, action):
        next_state, reward, done, _ = self.env_wrapper.step(VALID_ACTIONS[action])
        next_state = self.state_processor.process(next_state)
        # Why do we do this?
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

        return next_state, reward, done

    def run(self,
            episodes_to_run,
            update_target_estimator_every=10000,
            discount_factor=0.99,
            batch_size=32,
            save_model_every=25):

        # TODO: Shorten this function as much as possible
        for episode in range(episodes_to_run):

            if episode % save_model_every == 0:
                self.lib.save()

            state = self.reset_env()
            loss = None

            episode_reward = 0
            episode_length = 0

            for timestep in itertools.count():

                epsilon = self.epsilon_decay_schedule.next_epsilon(self.total_steps)
                self.log.log_epsilon(epsilon, self.total_steps)

                if self.total_steps % update_target_estimator_every == 0:
                    self.target_estimator.copy_parameters_from(self.q_estimator)

                self.log.log_step(timestep, self.total_steps, episode, episodes_to_run, loss)

                action = self.policy(state, epsilon)
                next_state, reward, done = self.step(state, action)

                self.replay_memory.append(Transition(state, action, reward, next_state, done))

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

                episode_reward += reward
                episode_length += 1
                self.total_steps += 1

                if done:
                    break

                state = next_state

            self.log.log_episode(episode, episodes_to_run, episode_length, episode_reward, self.total_steps)

        self.env_wrapper.monitor.close()


if __name__ == "__main__":
    env = gym.envs.make("Breakout-v0")
    base_dir = os.path.abspath("./experiments/{}/01".format(env.spec.id))
    tf_lib = TensorFlow(base_dir)
    with tf_lib:
        dqn = DeepQLearner(tf_lib, env, base_dir)
        dqn.run(10000)
