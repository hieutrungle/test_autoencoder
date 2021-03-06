import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import configargparse
import timeit
# from tensorflow.python import ipu


parser = configargparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10,
                        help="batch size. default=10")
args = parser.parse_args()


# Variables for model hyperparameters
num_classes = 10
epochs = 3
input_shape = (28, 28, 1)

strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
batch_size = args.batch_size * num_replicas

for k, v in args.__dict__.items():
    print(f"{k}: {v}")
print()

devices = tf.config.list_physical_devices()
print(devices)
print(f"Tennsorflow version: {tf.__version__}\n")

def load_data():
    # Load the MNIST dataset from keras.datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = keras.layers.Conv2D(filter, kernel_size=(1, 1), strides=2, padding="same")(x)
    # Layer 1
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(filter, kernel_size=(5, 5), padding="same"))(x)
    x = keras.layers.MaxPooling2D()(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(filter, kernel_size=(3, 3), padding="same"))(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    return x

def residual_scale(x, filter):
    # copy tensor to variable called x_skip
    x_skip = keras.layers.Conv2D(filter, kernel_size=(1, 1), strides=1, padding="same")(x)
    # Layer 1
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(filter, kernel_size=(3, 3), padding="same"))(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    return x

def deconvolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = keras.layers.Conv2DTranspose(filter, kernel_size=(1, 1), strides=2, padding="same")(x)
    # Layer 1
    for _ in range(3):
        x = tfa.layers.WeightNormalization(keras.layers.Conv2D(filter, kernel_size=(5, 5), padding="same"))(x)
    x = keras.layers.UpSampling2D()(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def make_model():
    encoder_input = keras.Input(shape=(28, 28, 1), name="img")
    x = convolutional_block(encoder_input,40)
    x = residual_scale(x,80)
    x = convolutional_block(x,130)
    x = residual_scale(x,160)
    encoder_output = tfa.layers.WeightNormalization(keras.layers.Conv2D(3, kernel_size=(3, 3), padding="same"))(x)

    x = keras.layers.Conv2DTranspose(160, 3, padding="same")(encoder_output)
    x = residual_scale(x,160)
    x = deconvolutional_block(x, 130)
    x = residual_scale(x,80)
    x = deconvolutional_block(x, 40)
    decoder_output = keras.layers.Conv2DTranspose(1, 3, padding="same")(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
    return autoencoder
    

def make_divisible(number, divisor):
    return number - number % divisor


# Prepare the dataset
(x_train, y_train), (x_test, y_test) = load_data()

data = np.vstack([x_train, x_test])
data_len = data.shape[0]
steps_per_execution = data_len // (batch_size * num_replicas)
steps_per_execution = make_divisible(steps_per_execution, num_replicas)
data_len = make_divisible(data_len, steps_per_execution * batch_size)
data= data[:data_len]
print(f"data: {data.shape}")
data = tf.convert_to_tensor(data, dtype=tf.float32)



with strategy.scope():
    model = make_model()
    model.summary()
    model.compile('adam', 'mse', steps_per_execution=steps_per_execution)
    start_time = timeit.default_timer()
    print('\nTraining')
    model.fit(data, data, epochs=epochs, batch_size=batch_size)
    training_time = timeit.default_timer()-start_time
    print(f"training time: {training_time}")