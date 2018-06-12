import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # x = tf.layers.dense(x, 64)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class RecurrentActor(Model):
    def __init__(self, nb_actions, name="actor", layer_norm=True):
        super(RecurrentActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, rnn_state_tuple, batch_size, step_size, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            lstm_cell = tc.rnn.LSTMCell(64, state_is_tuple=True)
            rnn_input = tf.reshape(x, [batch_size, step_size, x.shape[-1]])
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_input,
                initial_state=rnn_state_tuple,
                sequence_length=tf.fill([batch_size], 1),
                time_major=False
            )

            x = tf.reshape(lstm_outputs, [batch_size * step_size, 64])

            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)

        return x, lstm_state


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            # x = tf.layers.dense(x, 64)
            # if self.layer_norm:
            #     x = tc.layers.layer_norm(x, center=True, scale=True)
            # x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class RecurrentCritic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(RecurrentCritic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, rnn_state_tuple, batch_size, step_size, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            y = action
            y = tf.layers.dense(y, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(y, center=True, scale=True)
            y = tf.nn.relu(y)

            x = tf.concat([x, y], axis=-1)

            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True)
            lstm_cell = tc.rnn.LSTMCell(64, state_is_tuple=True)
            rnn_input = tf.reshape(x, [batch_size, step_size, x.shape[-1]])
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                rnn_input,
                initial_state=rnn_state_tuple,
                sequence_length=tf.fill([batch_size], step_size),
                time_major=False
            )
            x = tf.reshape(lstm_outputs, [batch_size * step_size, 64])
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x, lstm_state

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
