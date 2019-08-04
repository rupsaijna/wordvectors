# wordvectors
examples with wordvectors

##  python version
Python 3.7.3

## dependencies
pip install -r requirements.txt

## setup data
```bash
git clone https://github.com/rupsaijna/wordvectors.git
mkdir wordvectors/data
cd wordvectors/data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -zxvf aclImdb_v1.tar.gz
```

### Using GloVe word embedding model from [here (822MB)](http://nlp.stanford.edu/data/glove.6B.zip)

```bash  
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```


## Locally Trained Glove
```bash 
git clone https://github.com/rupsaijna/glove-python.git 
cd glove-python 
pip install glove-python
python3 glove-python/examples/example.py -c allreviews.txt -t 10 -p 4
```
[combine all imdb reviews into one text file, one review per line and save it as allreviews.txt]
The above code creates a glove.model and a corpus.model

### Accessing vectors from locally trained model
```python
from glove import Glove
gm=Glove.load('glove.model')
print (gm.most_similar('queen', number=10)) # 10 most similar words to 'queen'
print (gm.word_vectors[gm.dictionary['queen']]) # word vector representation of 'queen'
```
