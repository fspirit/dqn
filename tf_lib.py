import tensorflow as tf
import os
import common

from tf_estimator import TensorFlowEstimator
from tf_state_processor import TensorFlowStateProcessor


class TensorFlow(object):

    def __init__(self, base_dir):
        tf.reset_default_graph()
        self._set_up_dirs(base_dir)

        self.sess = tf.Session()

        tf.train.create_global_step()
        self._create_estimators()

        self._restore_or_init_vars()

    def _restore_or_init_vars(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            saver = tf.train.Saver()
            saver.restore(self.sess, latest_checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

    def _create_estimators(self):
        self.q_estimator = TensorFlowEstimator(self.sess, len(common.VALID_ACTIONS), scope="q")
        self.target_q_estimator = TensorFlowEstimator(self.sess, len(common.VALID_ACTIONS), scope="target_q")

    def _set_up_dirs(self, base_dir):
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        self.checkpoint_path_prefix = os.path.join(self.checkpoint_dir, "model")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def get_q_estimator(self):
        return self.q_estimator

    def get_target_estimator(self):
        return self.target_q_estimator

    def get_state_processor(self):
        return TensorFlowStateProcessor(self.sess)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.checkpoint_path_prefix, global_step=tf.train.get_global_step())

    def get_number_of_steps_done(self):
        return self.sess.run(tf.train.get_global_step())

    def __enter__(self):
        self.sess.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.__exit__(exc_type, exc_val, exc_tb)


if __name__ == "__main__":
    tf_lib = TensorFlow("./experiments/Breakout-v0")
    print(tf_lib.get_number_of_steps_done())
