import tensorflow as tf

def init_optimiser(optimiser_name: str, learning_rate: float = 0.001):
    """
    Initialize a TensorFlow optimizer from a string name.

    Args:
        optimiser_name (str): Name of the optimizer, e.g. 'Adam', 'SGD', 'RMSprop'.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.optimizers.Optimizer: TensorFlow optimizer instance.
    """
    name = optimiser_name.lower()
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate)
    elif name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate)
    elif name == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer name '{optimiser_name}'. "
                         "Supported: 'Adam', 'SGD', 'RMSprop', 'Adagrad'.")
