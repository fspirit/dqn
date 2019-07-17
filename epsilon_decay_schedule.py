import numpy as np


class EpsilonDecaySchedule:

    def __init__(self, start, end, steps):
        self.steps = steps
        self.epsilons = np.linspace(start, end, self.steps)

    def next_epsilon(self, steps_taken_so_far):
        return self.epsilons[min(steps_taken_so_far, self.steps-1)]

