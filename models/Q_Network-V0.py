import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Activation, BatchNormalization, Dropout, MaxPooling2D, Add # type: ignore
from tensorflow.keras.activations import gelu # type: ignore
from tensorflow.keras import models,regularizers # type: ignore
from keras.optimizers import Adam,SGD # type: ignore
from keras.optimizers.schedules import ExponentialDecay # type: ignore

class Q_Network:
    def __init__(self, input_shape, action_size, length_episodes, learning_rate, optimizer='adam'):
        """
        Initializes the Q-network for Deep Q-Learning.

        Args:
            input_shape (tuple): Shape of the input (stacked frames).
            action_size (int): Number of possible actions in the environment.
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for training.
        """
        tf.random.set_seed(42)  # Ensure reproducibility
        self.input_shape = input_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        '''
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate= learning_rate,
            decay_steps= length_episodes,
            decay_rate=0.999,
            staircase=True)
        '''
        self.optimizer = self._init_optimizer(optimizer)
        self.model = self.build_network()


    def _init_optimizer(self, optimiser):
        if isinstance(optimiser, str):
            if optimiser == 'Adam':
                self.optimiser = Adam(learning_rate=self.learning_rate)
            elif optimiser == 'SGD':
                self.optimiser = SGD(learning_rate=self.learning_rate)
            else:
                raise ValueError("Unsupported optimizer.")
        else:
            self.optimiser = optimiser

    def conv_block(self, inputs, filters, kernel_size, strides=1):
        """
        Residual convolutional block with 1x1 skip conv.
        """
        # First conv
        x = Conv2D(filters, kernel_size, strides=strides, padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = Activation(gelu)(x)

        # Second conv
        x = Conv2D(filters, kernel_size, padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(1e-4))(x)

        # Skip connection with 1x1 conv (if needed for shape match)
        skip = Conv2D(filters, (1, 1), strides=strides, padding='same')(inputs)

        # Residual add
        x = Add()([x, skip])
        x = Activation(gelu)(x)

        return x

    def build_network(self):
        """
        Builds Q-network (ResNet-inspired CNN).
        """
        inputs = Input(shape=self.input_shape)

        # Feature extractor
        block_1 = self.conv_block(inputs, 32, (8, 8), strides=4)
        block_2 = self.conv_block(block_1, 64, (4, 4), strides=2)
        block_3 = self.conv_block(block_2, 64, (3, 3), strides=1)

        # Flatten + FC
        x = Flatten()(block_3)
        x = Dense(512, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(1e-4))(x)
        x = Activation(gelu)(x)

        # Output Q-values
        outputs = Dense(self.action_size, activation='linear')(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss='huber')

        return model



    def predict(self, state_batch):
        """
        Predicts Q-values for a batch of states.

        Args:
            state_batch (np.ndarray): Batch of state inputs.

        Returns:
            np.ndarray: Predicted Q-values for each action.
        """
        return self.model(state_batch, training=False).numpy()

    def save(self, path):
        """
        Saves the model weights to the specified path.

        Args:
            path (str): File path to save weights.
        """
        self.model.save_weights(path)

    def load(self, path):
        """
        Loads the model weights from the specified path.

        Args:
            path (str): File path to load weights.
        """
        self.model.load_weights(path)

    def summary(self):
        """
        Prints a summary of the model architecture.
        """
        self.model.summary()
