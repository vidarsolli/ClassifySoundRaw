from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import json
import time
import sys, getopt
import librosa
import os
import random
import numpy as np
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



#labels = ['a_n', 'a_l','a_h', 'a_lhl','i_n', 'i_l','i_h', 'i_lhl', 'u_n', 'u_l','u_h', 'u_lhl',]
labels = ['a','i', 'u',]


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

# Build training set
audio_files = get_audiofiles(cp["test_folder"])
window_size = cp["window_size"]
step_size = cp["step_size"]
print("Window_size: ", window_size, "Step_size: ", step_size)
y_train = np.array([])
x_train = np.array([])
for file in audio_files:
    audio_samples, sample_rate = librosa.load(file)
    audio_samples=librosa.resample(audio_samples, sample_rate, cp["sample_rate"])
    no_of_examples = int((audio_samples.shape[0]-window_size)/step_size)-1
    # dt = time between each feature vector
    dt = step_size/sample_rate
    print("Extracting features from ", file, "# samples: ", audio_samples.shape, " sr: ", cp["sample_rate"], " dt: ", dt, "# features: ", no_of_examples)

    #for i in range(no_of_examples):
    y_vector = np.zeros(len(labels))
    label = str.split(file, '.')[1]
    label = str.split(label, "-")[1]
    label = str.split(label, "_")[0]
    y_vector[labels.index(label)] = 1
    for i in range(no_of_examples):
        y = audio_samples[(i*step_size):(i*step_size+window_size)]
        x_train = np.append(x_train, y)
        label = str.split(file, '.')[1]
        label = str.split(label, "-")[1]
        label = str.split(label, "_")[0]
        #y_train = np.append(y_train, labels.index(label))
        y_train = np.append(y_train, y_vector)

x_train = np.reshape(x_train, (int(len(x_train)/window_size), window_size, 1))
y_train = np.reshape(y_train, (int(len(y_train) / len(labels)), len(labels), 1))
#x_train += 1
#x_train *= 0.4

# Restore the model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


# Reading the output of the encoder layer
y_predict = np.array([])
print("Reading the encoder output")
#x_predict = np.reshape(x_predict, (1,800,1))
# x_predict = x_train[i:i+1][:][:]
#    intermediate_output = model.predict(x_predict)
y_predict = model.predict(x_train)
y_predict = np.reshape(y_predict, (y_train.shape[0],y_train.shape[1]))
print("Y-predict shape: ", y_predict.shape)
print("Y-train shape: ", y_train.shape)
#print("Y-train: ", y_train[i], "  Y-predict: ", y_predict[i])
# model.predict(x_predict, batch_size=1)
# layer_value = model.layers[3].output.eval(session= K.get_session())
# print(type(intermediate_output), intermediate_output.shape)
#x_cluster = np.append(x_cluster, model.layers[3].output)

# Print the confusion matrix

#Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
#y_pred = np.argmax(Y_pred, axis=1)
#x = np.where(y_predict[:, 0:y_predict.shape[1]] == np.amax(y_predict[:, 0:y_predict.shape[1]]))
x = np.zeros(y_predict.shape)
xc = np.zeros([y_predict.shape[0]])
y = np.zeros(y_predict.shape)
yc = np.zeros([y_predict.shape[0]])
for i in range(y_predict.shape[0]):
    m = np.amax(y_predict[i, :])
    x[i,np.where(y_predict[i, :] == m)] = 1
    xc[i] = np.where(y_predict[i, :] == m)[0]
    m = np.amax(y_train[i, :])
    y[i,np.where(y_train[i, :] == m)] = 1
    yc[i] = np.where(y_train[i, :] == m)[0]
print('Classification Report')
print(classification_report(x, y, target_names=labels))
print('Confusion Matrix')
print(confusion_matrix(xc, yc))