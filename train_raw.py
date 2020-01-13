from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, BatchNormalization
from keras.models import Model
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import json
import time
import sys, getopt
import librosa
import os
import random
import matplotlib.pyplot as plt
import os.path
from os import path


import numpy as np

#labels = ['a_n', 'a_l','a_h', 'a_lhl','i_n', 'i_l','i_h', 'i_lhl', 'u_n', 'u_l','u_h', 'u_lhl',]
labels = ['a', 'i', 'u',]


"""
AutodecoderRaw will train a 1D convolutional autoencoder with raw audio signals
and cluster the encoder vector to see if clusters contains sound with similar characteristics.
The parameters for the network are stored in a .json file
The audiofiles to be used are stored in a folder (folder to be stated in the .json file).
Usage:  python3 AutoencoderRaw -i <filenam>.json
"""

json_file = "./config.json"

# Return a list of audio files
def get_audiofiles(folder):
    list_of_audio = []
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (folder, file)
            list_of_audio.append(directory)
    return list_of_audio


# Get the input json file
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

window_size = cp["window_size"]
step_size = cp["step_size"]
print("Window_size: ", window_size, "Step_size: ", step_size)
if path.exists("x_train.npy") and path.exists("y_train.npy"):
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
else:
    # Build training set
    audio_files = get_audiofiles(cp["audio_folder"])
    y_train = np.array([])
    x_train = np.array([])
    for file in audio_files:
        audio_samples, sample_rate = librosa.load(file)
        audio_samples=librosa.resample(audio_samples, sample_rate, cp["sample_rate"])
        no_of_examples = int((audio_samples.shape[0]-window_size)/step_size)-1
        dt = step_size/sample_rate
        print("Extracting features from ", file, "# samples: ", audio_samples.shape, " sr: ", cp["sample_rate"], " dt: ", dt, "# examples: ", no_of_examples)

        for i in range(no_of_examples):
            y = audio_samples[(i*step_size):(i*step_size+window_size)]
            x_train = np.append(x_train, y)
            y_vector = np.zeros(len(labels))
            label = str.split(file, '.')[1]
            label = str.split(label, "-")[1]
            label = str.split(label, "_")[0]
            y_vector[labels.index(label)] = 1
            y_train = np.append(y_train, y_vector)

    x_train = np.reshape(x_train, (int(len(x_train)/window_size), window_size, 1))
    y_train = np.reshape(y_train, (int(len(y_train)/len(labels)), len(labels), 1))
    #x_train += 1
    #x_train *= 0.4
    min = np.min(x_train)
    max = np.max(x_train)
    print(min, max)
    print("X_train shape: ", x_train.shape)
    np.save("x_train", x_train)
    print("Y_train shape: ", y_train.shape)
    np.save("y_train", y_train)

# Set up the SampleCNN model

input = Input(shape=(window_size, 1))  # adapt this if using `channels_first` image data format
print("Input shape: ", input.shape)

x = Conv1D(cp["no_of_filter1"], cp["filter_size"], activation=cp["activation"], strides=3, data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(input)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Conv1D(cp["no_of_filter2"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(x)
x = Dropout(0.2)(x)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Conv1D(cp["no_of_filter3"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(x)
x = Dropout(0.2)(x)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Conv1D(cp["no_of_filter4"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(x)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Conv1D(cp["no_of_filter5"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(x)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Conv1D(cp["no_of_filter6"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(x)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Conv1D(cp["no_of_filter7"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same', kernel_constraint=maxnorm(3))(x)
if cp["batch_norm"]:
    x = BatchNormalization()(x)
x = MaxPooling1D(cp["filter_size"], padding='same')(x)
#x = Conv1D(cp["no_of_filter8"], cp["filter_size"], activation=cp["activation"], strides=cp["stride"], data_format='channels_last', padding='same')(x)
#x = MaxPooling1D(cp["filter_size"], padding='same')(x)
x = Dense(cp["dense1_size"], activation=cp["activation"])(x)
decoded = Dense(len(labels), activation='softmax') (x)

print(type(decoded))

autoencoder = Model(input, decoded)

plot_model(autoencoder, show_shapes=True, expand_nested=True, to_file='model.png')
print("Compiling model")
sgd = SGD(lr=0.005, momentum=0.8)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(autoencoder.summary())

# Start the TensorBoard server by: tensorboard --logdir=/tmp/autoencoder
# and navigate to: http://0.0.0.0:6006
# In case, add: callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]

print(x_train.shape)

earlystopper = EarlyStopping(monitor='loss', min_delta=0.000001, patience=10, verbose=1)

y_train = np.reshape(y_train, (y_train.shape[0], 1, len(labels)))

if cp["train"]:
    history = autoencoder.fit(x_train, y_train,
                epochs=cp["epochs"],
                batch_size=cp["batch_size"],
                verbose=1,
                validation_split=cp["validation_split"],
                callbacks=[earlystopper])

# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")

# Plot the result
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()