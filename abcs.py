import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import timeit
import configargparse
from tensorflow.python import ipu

parser = configargparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10,
                        help="batch size. default=32")
parser.add_argument('--num_ipus', type=int, default=1,
                        help="num_ipus")
args = parser.parse_args()


for k, v in args.__dict__.items():
    print(f"{k}: {v}")
print()

devices = tf.config.list_physical_devices()
print(devices)
print(f"Tennsorflow version: {tf.__version__}\n")


# Variables for model hyperparameters
num_classes = 10
input_shape = (28, 28, 1)
epochs = 100
batch_size = args.batch_size
num_ipus = num_replicas = args.num_ipus

print(f"epochs: {epochs}")

# Load the MNIST dataset from keras.datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


def make_divisible(number, divisor):
    return number - number % divisor

# Adjust dataset lengths to be divisible by the batch size
train_data_len = x_train.shape[0]
train_steps_per_execution = train_data_len // (batch_size * num_replicas)
# `steps_per_execution` needs to be divisible by the number of replicas
train_steps_per_execution = make_divisible(train_steps_per_execution, num_replicas)
train_data_len = make_divisible(train_data_len, train_steps_per_execution * batch_size)
x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

test_data_len = x_test.shape[0]
test_steps_per_execution = test_data_len // (batch_size * num_replicas)
# `steps_per_execution` needs to be divisible by the number of replicas
test_steps_per_execution = make_divisible(test_steps_per_execution, num_replicas)
test_data_len = make_divisible(test_data_len, test_steps_per_execution * batch_size)
x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]

# Normalize the images.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# When dealing with images, we usually want an explicit channel dimension, even when it is 1.
# Each sample thus has a shape of (28, 28, 1).
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Finally, convert class assignments to a binary class matrix.
# Each row can be seen as a rank-1 "one-hot" tensor.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def model_fn():
    # Input layer - "entry point" / "source vertex".
    input_layer = keras.Input(shape=input_shape)
    x = input_layer
    # Add layers to the graph.
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))(x)
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu"))(x)
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(128, kernel_size=(5, 5), padding="same", activation="relu"))(x)


    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu"))(x)
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(128, kernel_size=(5, 5), padding="same", activation="relu"))(x)
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))(x)
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))(x)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return input_layer, x


ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = num_ipus
ipu_config.configure_ipu_system()

# Model.__init__ takes two required arguments, inputs and outputs.

strategy = ipu.ipu_strategy.IPUStrategy()

with strategy.scope():
    model = keras.Model(*model_fn())

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    model.compile('sgd', 'categorical_crossentropy',
                  metrics=["accuracy"],
                  steps_per_execution=train_steps_per_execution)
    model.summary()
    start_time = timeit.default_timer()
    print('\nTraining')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    training_time = timeit.default_timer()-start_time
    print(f"training time: {training_time}")

print("Program ran successfully")
