import numpy as np


def make_epsilon_greedy_policy(estimator, actions_count):

    def policy_fn(observation, next_eps_fn):
        epsilon = next_eps_fn()
        A = np.ones(actions_count, dtype=float) * epsilon / actions_count
        q_values = estimator.predict(np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return np.random.choice(len(A), p=A)

    return policy_fn
