#importing the glove library
from glove import Corpus, Glove

import tqdm, re, os, numpy as np 

#
# Configuration
#
MAX_NB_WORDS=25000 #top nb_words most common words
MAX_SEQUENCE_LENGTH=1000 #max length of review
N_GLOVE_TOKENS=400000
EMBEDDING_DIM = 100
NUMREV = 100 #number of reviews, total set: NUMREV positive + NUMREV negative


#
# PREPROCESSING
#
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]") # removes these symbols (piunctuation marks)
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)") #changes tags , - , / with a space to maintain sense of a sentence.
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

# function which returns text ( lowercased ) from a file
def read_text(filename):
        with open(filename) as f:
                return f.read().lower()

print ("\nReading negative reviews.")
negative_text = [read_text(os.path.join(negative_dir, filename))
        for filename in tqdm.tqdm(os.listdir(negative_dir))]

""" print(negative_text[0])
print("\n")
print( preprocess_reviews( [ negative_text[0] ]  ) )
raise SystemExit """

print ("\nReading positive reviews.")
positive_text = [read_text(os.path.join(positive_dir, filename))
        for filename in tqdm.tqdm(os.listdir(positive_dir))]



lines = preprocess_reviews(negative_text + positive_text)

print("lines from 1 to 3 : ", lines[1:3] )

# creating a corpus object
corpus = Corpus() 
#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(lines, window=10)
#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=5, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')
