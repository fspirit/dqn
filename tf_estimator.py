import tensorflow as tf
import os

class TensorFlowEstimator(object):
    def __init__(self, sess, actions_count, tf_board_logger=None, scope="estimator"):
        self.sess = sess
        self.scope = scope
        self.tf_board_logger = tf_board_logger

        with tf.variable_scope(self.scope):
            self._build_model(actions_count)
            self.sess.run(tf.global_variables_initializer())


    def _build_model(self, actions_count):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.layers.conv2d(X, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)

        # Fully connected layers
        flattened = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(flattened, 512)
        self.predictions = tf.layers.dense(fc1, actions_count)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard
        # TODO: Move this stuff to TFBoard object
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, s):
        return self.sess.run(self.predictions, {self.X_pl: s})

    def update(self, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = self.sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)

        if self.tf_board_logger:
            self.tf_board_logger.log_loss_and_q_values(summaries, global_step)

        return loss

    def copy_parameters_from(self, other_estimator):
        this_estimator_params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        this_estimator_params = sorted(this_estimator_params, key=lambda v: v.name)
        other_estimator_params = [t for t in tf.trainable_variables() if t.name.startswith(other_estimator.scope)]
        other_estimator_params = sorted(other_estimator_params, key=lambda v: v.name)

        update_ops = []
        for this_v, other_v in zip(this_estimator_params, other_estimator_params):
            op = this_v.assign(other_v)
            update_ops.append(op)

        self.sess.run(update_ops)

        print("\nCopied estimator parameters from '{0}' to '{1}'.\n".format(other_estimator.scope, self.scope))
