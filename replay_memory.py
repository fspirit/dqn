import numpy as np
import random
from common import *


class ReplayMemory:

    def __init__(self):
        self.buffer = []
        self.replay_memory_init_size = 50000
        self.replay_memory_size = 500000

    def append(self, transition):
        if len(self.buffer) == self.replay_memory_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    # TODO: Simililar to waht we do in DQLearner
    def reset_state(self, env, state_processor):
        state = env.reset()
        state = state_processor.process(state)
        state = np.stack([state] * 4, axis=2)

        return state

    def init_replay_memory(self,
                           env,
                           policy,
                           state_processor,
                           epsilon_decay_schedule,
                           total_steps):

        print("Populating replay memory...")

        state = self.reset_state(env, state_processor)

        for i in range(self.replay_memory_init_size):

            action = policy(state, epsilon_decay_schedule.next_epsilon(total_steps))

            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            self.append(Transition(state, action, reward, next_state, done))

            if done:
                state = self.reset_state(env, state_processor)
            else:
                state = next_state

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)
