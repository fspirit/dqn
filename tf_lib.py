import tensorflow as tf
import os
import common

from tf_estimator import TensorFlowEstimator
from tf_state_processor import TensorFlowStateProcessor

class TensorFlow:
    def __init__(self, working_dir):
        tf.reset_default_graph()

        self.working_dir = working_dir
        self.checkpoint_dir = os.path.join(working_dir, "checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _load_estimator(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver = tf.train.Saver()
            saver.restore(self.sess, latest_checkpoint)

    def get_q_estimator(self, logger):
        self._load_estimator()
        return TensorFlowEstimator(self.sess, len(common.VALID_ACTIONS), logger, scope="q")

    def get_target_estimator(self):
        return TensorFlowEstimator(self.sess, len(common.VALID_ACTIONS), scope="target_q")

    def get_state_processor(self):
        return TensorFlowStateProcessor(self.sess)

    def save_estimator(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.checkpoint_path)

    def get_number_of_steps_done(self):
        return self.sess.run(tf.contrib.framework.get_global_step())

    def __enter__(self):
        self.sess.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.__exit__(exc_type, exc_val, exc_tb)
