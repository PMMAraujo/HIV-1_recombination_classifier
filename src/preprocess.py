import pandas as pd
import numpy as np

from nltk import ngrams
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle

import logging
log = logging.getLogger(__name__)

def create_trigrams(input_data, is_df=False):
    log.info("Creating trigrams from genomes")

    if is_df == True:
        df = input_data
    else:
        df = pd.read_csv(input_data)

    sequences = df['seqs'].values
    
    corpus = []
    for seq in sequences:
        in_tri = np.array([''.join(i) for i in ngrams(seq, 3)])
        corpus.append(in_tri)

    sentences = [' '.join(x) for x in corpus]

    y = df['recomb'].values

    return sentences, y

def trigrams_tokenizer(sentences_train, sentences_test, num_words):
    log.info("Creating tokens from trigrams")

    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1

    return X_train, X_test, vocab_size, tokenizer

def padding(X_train, X_test, maxlen):
    log.info("Padding")

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_test