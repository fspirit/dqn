class Log(object):

    def __init__(self):
        self.loggers = []

    def add_logger(self, logger):
        self.loggers.append(logger)

    def log_step(self, timestep, total_steps, episode, total_episodes_count, loss):
        for logger in self.loggers:
            try:
                logger.log_step(timestep, total_steps, episode, total_episodes_count, loss)
            except AttributeError:
                pass

    def log_epsilon(self, epsilon, global_step):
        for logger in self.loggers:
            try:
                logger.log_epsilon(epsilon, global_step)
            except AttributeError:
                pass

    def log_episode(self, episode, total_episodes_count, episode_length, episode_reward, total_steps):
        for logger in self.loggers:
            try:
                logger.log_episode(episode, total_episodes_count, episode_length, episode_reward, total_steps)
            except AttributeError:
                pass
