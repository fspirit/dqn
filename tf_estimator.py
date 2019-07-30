import tensorflow as tf


class TensorFlowEstimator(object):
    def __init__(self, sess, actions_count, scope="estimator"):
        self.sess = sess
        self.scope = scope

        with tf.variable_scope(self.scope):
            self._build_model(actions_count)

    def _build_model(self, actions_count):
        # Placeholders for our input
        # Our input are 4 RGB frames of shape 84, 84 each
        self.input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.target_q_for_selected_actions = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.selected_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.input) / 255.0
        batch_size = tf.shape(self.input)[0]

        # Three convolutional layers
        conv1 = tf.layers.conv2d(X, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)

        # Fully connected layers
        flattened = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(flattened, 512)
        self.q_values_for_all_actions = tf.layers.dense(fc1, actions_count)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.q_values_for_all_actions)[1] + self.selected_actions
        self.q_values_for_selected_actions = tf.gather(tf.reshape(self.q_values_for_all_actions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.target_q_for_selected_actions, self.q_values_for_selected_actions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, s):
        return self.sess.run(self.q_values_for_all_actions, {self.input: s})

    def update(self, s, a, y):
        feed_dict = {self.input: s, self.target_q_for_selected_actions: y, self.selected_actions: a}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)

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

        print("\nCopied estimator parameters from '{0}' to '{1}'.".format(other_estimator.scope, self.scope))
