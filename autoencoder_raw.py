from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys, getopt
from os import path
import time

np.random.seed(1337)

# Read the configuration parameters from a .json file
json_file = "config.json"
try:
    myOpts, args = getopt.getopt(sys.argv[1:], "i:")
except getopt.GetoptError as e:
    print(str(e))
    print("Usage: %s -i <json_file>" % sys.argv[0])
    sys.exit(2)

for o, a in myOpts:
    if o == '-i':
        json_file = a

with open(json_file) as file:
    cp = json.load(file)

if path.exists("x_train.npy") and path.exists("y_train.npy"):
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
else:
    print("No training file exists")
    exit()

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
max = np.max(x_train)
min = np.min(x_train)
x_train = x_train - min
x_train = x_train / max

print(x_train.shape)
# Network parameters
input_shape = (cp["window_size"], 1, 1)
batch_size = cp["batch_size"]
kernel_size = 3
latent_dim = 16
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [cp["no_of_filter1"], cp["no_of_filter2"], cp["no_of_filter3"], cp["no_of_filter4"],
                 cp["no_of_filter5"], cp["no_of_filter6"], cp["no_of_filter7"], cp["no_of_filter8"]]


# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
no_of_layers = 0
rest = cp["window_size"]
for i in range(10):
    rest = rest/3
    if rest == 1.:
        no_of_layers = i
print("No of layers: ", no_of_layers)

x = Conv2D(cp["no_of_filter1"], cp["filter_size"], activation=cp["activation"], strides=(cp["first_stride"], 1), data_format='channels_last', padding='same')(inputs)
x = MaxPooling2D(cp["filter_size"], padding='same')(x)
x = Conv2D(cp["no_of_filter2"], cp["filter_size"], activation=cp["activation"], strides=(cp["stride"], 1), data_format='channels_last', padding='same')(x)
x = MaxPooling2D(cp["filter_size"], padding='same')(x)
x = Conv2D(cp["no_of_filter3"], cp["filter_size"], activation=cp["activation"], strides=(cp["stride"], 1), data_format='channels_last', padding='same')(x)
x = MaxPooling2D(cp["filter_size"], padding='same')(x)
if no_of_layers >= 4:
    x = Conv2D(cp["no_of_filter4"], cp["filter_size"], activation=cp["activation"], strides=(cp["stride"], 1), data_format='channels_last', padding='same')(x)
    x = MaxPooling2D(cp["filter_size"], padding='same')(x)
if no_of_layers >= 5:
    x = Conv2D(cp["no_of_filter5"], cp["filter_size"], activation=cp["activation"], strides=(cp["stride"], 1), data_format='channels_last', padding='same')(x)
    x = MaxPooling2D(cp["filter_size"], padding='same')(x)
if no_of_layers >= 6:
    x = Conv2D(cp["no_of_filter6"], cp["filter_size"], activation=cp["activation"], strides=(cp["stride"], 1), data_format='channels_last', padding='same')(x)
    x = MaxPooling2D(cp["filter_size"], padding='same')(x)
if no_of_layers >= 7:
    x = Conv2D(cp["no_of_filter7"], cp["filter_size"], activation=cp["activation"], strides=(cp["stride"], 1), data_format='channels_last', padding='same')(x)
    x = MaxPooling2D(cp["filter_size"], padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of Transposed Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use UpSampling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=(3,1),
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation(cp["last_activation"], name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

sgd = SGD(lr=0.1, momentum=0.8)

autoencoder.compile(optimizer='adam', loss=cp["loss_function"], metrics=['accuracy'])
# Train the autoencoder
history = autoencoder.fit(x_train,
                x_train,
                #validation_data=(x_test_noisy, x_test),
                epochs=10,
                validation_split = 0.2,
                batch_size=cp["batch_size"])

# Predict the Autoencoder output from corrupted test images
x_decoded = autoencoder.predict(x_train)

# Plot the result
file_time = time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime())
# Plot training & validation loss values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Autoencoder accuracy and loss')
plt.ylabel('Accuracy/Loss')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Test accuracy', "Train loss", "Test loss"], loc='upper left')
plt.savefig(file_time+"autoencoder.png", format="png")
plt.show()
