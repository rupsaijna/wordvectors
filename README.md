# wordvectors
examples with wordvectors

## dependencies
numpy

scipy

sklearn

keras

tqdm (for pretty print)

## setup data
git clone https://github.com/rupsaijna/wordvectors.git

mkdir wordvectors/data

cd wordvectors/data

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

tar -zxvf aclImdb_v1.tar.gz


### Using GloVe word embedding model from [here (822MB)](http://nlp.stanford.edu/data/glove.6B.zip)

wget http://nlp.stanford.edu/data/glove.6B.zip

unzip glove.6B.zip

