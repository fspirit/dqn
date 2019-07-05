import os
import tensorflow as tf


class TFBoardLogger(object):

    def __init__(self, scope, working_dir="."):
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            if working_dir:
                summary_dir = os.path.join(working_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def log_epsilon(self, epsilon, global_step):
        summary = tf.Summary()
        summary.value.add(simple_value=epsilon, tag="epsilon")
        self.summary_writer.add_summary(summary, global_step)

    def log_episode_stats(self, reward, length, global_step):
        summary = tf.Summary()
        summary.value.add(simple_value=reward, node_name="episode_reward", tag="episode_reward")
        summary.value.add(simple_value=length, node_name="episode_length", tag="episode_length")
        self.summary_writer.add_summary(summary, global_step)

    def log_loss_and_q_values(self, summaries, global_step):
        self.summary_writer.add_summary(summaries, global_step)
