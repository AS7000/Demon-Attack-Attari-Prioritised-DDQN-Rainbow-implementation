import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Activation, Lambda #type:ignore
from tensorflow.keras import models, regularizers # type: ignore
from tensorflow.keras.losses import Huber # type: ignore

class Q_Network:
    """
    Canonical Dueling DQN convolutional network for Atari (channels-last).
    Meant to be used with Prioritized Replay + Double DQN logic externally.
    """

    def __init__(self, input_shape=(84, 84, 4), action_size=6, learning_rate=1e-4, optimizer='adam', clipnorm=10.0):
        """
        Args:
            input_shape: tuple, e.g. (84,84,4)
            action_size: int, number of discrete actions
            learning_rate: float
            optimizer: 'adam'|'sgd' or a tf.keras.optimizers.Optimizer instance
            clipnorm: float, gradient clipping by norm (helps stability)
        """
        tf.random.set_seed(42)
        self.input_shape = input_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.optimizer = self._init_optimizer(optimizer)
        self.model = self.build_network()
        # Compile with Huber loss (common in DQN)
        self.model.compile(optimizer=self.optimizer, loss=Huber())

    def _init_optimizer(self, optimiser):
        if isinstance(optimiser, str):
            if optimiser.lower() == 'adam':
                return tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
            elif optimiser.lower() == 'sgd':
                return tf.keras.optimizers.SGD(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
            else:
                raise ValueError("Unsupported optimizer string. Use 'adam' or 'sgd' or pass an optimizer instance.")
        else:
            # assume user passed an optimizer instance; set clipnorm if possible
            try:
                optimiser.clipnorm = self.clipnorm
            except Exception:
                pass
            return optimiser

    def build_network(self):
        """
        Build the standard DQN conv backbone + dueling head (channels-last).
        """
        inputs = Input(shape=self.input_shape, name='state_input')  # (84,84,4)

        # Classic DQN conv-stack (no residuals) -> matches original Atari DQN
        # Conv1: 32 filters, 8x8 kernel, stride 4
        x = Conv2D(32, kernel_size=8, strides=4, padding='valid',
                   activation='relu', kernel_initializer='he_normal')(inputs)

        # Conv2: 64 filters, 4x4 kernel, stride 2
        x = Conv2D(64, kernel_size=4, strides=2, padding='valid',
                   activation='relu', kernel_initializer='he_normal')(x)

        # Conv3: 64 filters, 3x3 kernel, stride 1
        x = Conv2D(64, kernel_size=3, strides=1, padding='valid',
                   activation='relu', kernel_initializer='he_normal')(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))(x)

        # Dueling streams
        # Value stream
        v = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        v = Dense(1, activation='linear', name='value')(v)  # V(s)

        # Advantage stream
        a = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        a = Dense(self.action_size, activation='linear', name='advantage')(a)  # A(s,a)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        def combine_va(tensors):
            v_t, a_t = tensors
            a_mean = tf.reduce_mean(a_t, axis=1, keepdims=True)  # shape (batch,1)
            q = v_t + (a_t - a_mean)
            return q

        q_values = Lambda(combine_va, name='q_values')([v, a])

        model = models.Model(inputs=inputs, outputs=q_values, name='DuelingDQN_CNN')
        return model

    def predict(self, state_batch):
        """
        Predict Q-values. state_batch shape: (batch, 84,84,4)
        """
        return self.model(state_batch, training=False).numpy()

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def summary(self):
        self.model.summary()
