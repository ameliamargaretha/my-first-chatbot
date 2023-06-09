# source https://github.com/kunci115/siPintar

import nltk
import pickle
import numpy as np
from xgboost import train
import tflearn
import tensorflow as tf
import random 
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

data_file = open('intents2.json').read()
intents = json.loads(data_file)

words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem, lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print(documents, 'documents')
# classes = intents
print(classes, 'classes')
# words = all words, vocab
print('unique stemmed words', words)

# create training data
training = []
output = []

# create empty array for output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.compat.v1.reset_default_graph()

# build neural network
input_h = tflearn.input_data(shape=(None, len(train_x[0])))
h2 = tflearn.fully_connected(input_h, 9)
h3 = tflearn.fully_connected(h2, 18)
h4 = tflearn.fully_connected(h3, 18)
h5 = tflearn.fully_connected(h4, 9)

output_h = tflearn.fully_connected(h5, len(train_y[0]), activation='softmax')
output_h_reg = tflearn.regression(output_h)

# define model and setup tensorboard
model = tflearn.DNN(output_h_reg, tensorboard_dir = 'tflearn_logs')

# start training with Gradient Descent algorithm
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
hist = model.fit(train_x, train_y, n_epoch=3000, batch_size=10, show_metric=True)
model.save('chatbot2_model.h5')

pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data', 'wb'))