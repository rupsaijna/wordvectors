'''
Dataset: IMDB
Algo : 1D Convolutional NN, Maxpooling, RELU activation, 2 Dense Layer
loss Categorical_crossentropy,ADAM optimizer
Metric Accuracy
'''

import re
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv1D, Dense, Embedding, Flatten, Input, LSTM, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time
#
# Configuration
#
MAX_NB_WORDS=25000 #top nb_words most common words
MAX_SEQUENCE_LENGTH=1000 #max length of review
N_GLOVE_TOKENS=400000
EMBEDDING_DIM = 100
NUMREV = 100 #number of reviews, total set: NUMREV positive + NUMREV negative

#
#PREPROCESSING
#
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

#
# Load the data
#
positive_dir = "../data/aclImdb/train/pos"
negative_dir = "../data/aclImdb/train/neg"
glove_file="../data/glove/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

def read_text(filename):
        with open(filename) as f:
                return f.read().lower()

print ("\nReading negative reviews.")
negative_text = [read_text(os.path.join(negative_dir, filename))
        for filename in tqdm.tqdm(os.listdir(negative_dir))]
        
print ("\nReading positive reviews.")
positive_text = [read_text(os.path.join(positive_dir, filename))
        for filename in tqdm.tqdm(os.listdir(positive_dir))]


labels_index = { "negative": 0, "positive": 1 }

labels = [0 for _ in range(NUMREV)] + [1 for _ in range(NUMREV)]

texts = preprocess_reviews(negative_text[:NUMREV]) + preprocess_reviews(positive_text[:NUMREV])

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
#print reverse_word_map

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) #all equal!
labels = np_utils.to_categorical(np.asarray(labels))
print ("data.shape = {0}, labels.shape = {1}".format(data.shape, labels.shape))

x_train, x_test, y_train, y_test = train_test_split(data, labels)


#
# Load word embeddings
#
print("\nLoading word embeddings.")
embeddings_index = dict()
with open(glove_file) as f:
        for line in tqdm.tqdm(f, total=N_GLOVE_TOKENS):
                values = line.split()
                word, coefficients = values[0], np.asarray(values[1:], dtype=np.float32)
                embeddings_index[word] = coefficients

embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

print ("\nEmbedding_matrix.shape = {0}".format(embedding_matrix.shape))
print('\n')
embedding_layer = Embedding(len(word_index)+1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation="relu")(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)

preds = Dense(len(labels_index), activation="softmax")(x)

model = Model(sequence_input, preds)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])

model.summary()


start = time.clock()
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
end = time.clock()
print('Time spent:', end-start)

score = model.evaluate(x_test, y_test)
print('\nTraining Acc: %.2f%%' %(score[0]*100), '\nTest Acc: %.2f%%' %(score[1]*100))
