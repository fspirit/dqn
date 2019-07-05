from collections import namedtuple

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])