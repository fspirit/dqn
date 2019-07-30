import os
import tensorflow as tf


class TFBoardLogger(object):

    def __init__(self, base_dir="."):
        summary_dir = os.path.join(base_dir, "summaries")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir)

    def log_epsilon(self, epsilon, total_steps):
        summary = tf.Summary()
        summary.value.add(simple_value=epsilon, tag="epsilon")
        self.summary_writer.add_summary(summary, total_steps)

    def log_episode(self, episode, total_episodes, episode_length, episode_reward, total_steps):
        summary = tf.Summary()
        summary.value.add(simple_value=episode_length, node_name="episode_length", tag="episode_length")
        summary.value.add(simple_value=episode_reward, node_name="episode_reward", tag="episode_reward")

        # TODO: What is total_steps here? Can we replace it by episode?
        self.summary_writer.add_summary(summary, total_steps)


