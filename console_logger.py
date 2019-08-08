import sys


class ConsoleLogger(object):

    def log_step(self, timestep, total_steps, episode, total_episodes, loss):
        print("\rStep {} ({}) @ Episode {}/{}, Loss: {}".format(
            timestep + 1, total_steps, episode + 1, total_episodes, loss), end="")
        sys.stdout.flush()

    def log_episode(self, episode, total_episodes, episode_length, episode_reward, total_steps):
        print("\nEpisode {}/{}, Length: {}, Reward: {}".format(episode + 1, total_episodes,
                                                               episode_length, episode_reward))
