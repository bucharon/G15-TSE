import numpy
import sys
import nltk
import time
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K
import tensorflow as tf
 
cpus_input = 0
while cpus_input not in range(1, 9):
    cpus_input = int(input("How many CPU cores do you want to use for training? (1-8): "))
    if cpus_input not in range(1, 9):
        print("Please choose a value between 1 and 8")
 
print("Configuring training session with " + str(cpus_input) + " CPUs...")
config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': cpus_input})
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
 
nltk.download('stopwords')
file = open("ChildrenStoriesSingleRow2.csv", encoding='utf-8').read()
 
 
def count_spaces(file_input):
    spaces = 0
    for c in file_input:
        if c == ' ':
            spaces += 1
    return spaces
 
 
# Sleep program for 2 seconds so keywords input prompt isn't printed between TensorFlow libraries being opened
print("Please wait...")
time.sleep(2)
keywords_input = input("Choose keywords to train the neural network with, separated by spaces (or press ENTER to "
                       "skip this step): ")
keywords_input_array = keywords_input.split(" ")
print("Total keywords: " + str(len(keywords_input_array)))
 
 
def normalise_keywords(keywords_array, file_input):
    new_array = []
    # Check keywords input array has length > 0 to avoid division by 0 error
    if len(keywords_array) > 0:
        # Calculate prevalence of each word
        prevalence = count_spaces(file_input) // (4 * len(keywords_array))
        print("Keyword prevalence: " + str(prevalence))
        # Set counter for current index in new array
        for m in range(0, len(keywords_array)):
            for n in range(0, prevalence):
                new_array.append(keywords_array[m])
    # Return normalised keywords array
    return new_array
 
 
# Concatenate keywords to file to be tokenized
file = file + " " + " ".join(normalise_keywords(keywords_input_array, file))
 
 
# User inputs # of epochs, can only be between 1 and 100
epochs_input = 0
while not(0 < epochs_input < 101):
    epochs_input = int(input("Enter number of epochs for training (max 100, one epoch can take 1-3 minutes depending "
                             "on dataset and number of keywords): "))
    if not(0 < epochs_input < 101):
        print("Please enter a number between 1 and 100.")
 
 
# Validation for batch size input, start by creating array of 2^n for n in range 1 to 8 inclusive
valid_batch_sizes = [2**i for i in range(1, 9)]
batch_input = 0
while batch_input not in valid_batch_sizes:
    batch_input = int(input("Enter batch size (must be a power of 2 between 2 and 256, smaller batch size may "
                            "lengthen training time): "))
    if batch_input not in valid_batch_sizes:
        print("Please enter a valid batch size ( " + str(valid_batch_sizes) + "): ")
 
 
def tokenize_words(text_input):
    text_input = text_input.lower()
 
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text_input)
 
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)
 
 
processed_inputs = tokenize_words(file)
 
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))
 
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:", input_len)
print("Total vocab:", vocab_len)
 
seq_length = 100
x_data = []
y_data = []
 
# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]
 
    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]
 
    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
 
n_patterns = len(x_data)
print("Total Patterns:", n_patterns)
 
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X / float(vocab_len)
 
y = np_utils.to_categorical(y_data)
 
# Sequential Keras model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='adam')
 
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]
 
print("Fitting model with " + str(epochs_input) + " epochs and batch size " + str(batch_input) + ".")
model.fit(X, y, epochs=epochs_input, batch_size=batch_input, callbacks=desired_callbacks)
 
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
 
num_to_char = dict((i, c) for i, c in enumerate(chars))
 
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
 
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
 
    sys.stdout.write(result)
 
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
