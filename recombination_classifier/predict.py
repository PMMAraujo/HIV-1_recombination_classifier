import sys
sys.path.append(".")
from argparse import ArgumentParser

import numpy as np
import pickle

from nltk import ngrams
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from recombination_classifier import models

VOCAB_SIZE = 1665
NUM_WORDS = 5000
MAXLEN = 3000
EMBEDDING_DIM = 50


def create_tokens(NUM_WORDS, ngrams):
    with open('./recombination_classifier/models_files/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenized = tokenizer.texts_to_sequences(ngrams)
    return tokenized

def create_model():
    model = models.create_model(VOCAB_SIZE, EMBEDDING_DIM, MAXLEN)
    model.load_weights('./recombination_classifier/models_files/model.weights')

    return model

def make_pred(string_input):

    # from string to trigrams
    in_tri = np.array([''.join(i) for i in ngrams(string_input, 3)])

    # tokenizing
    as_tokens = create_tokens(NUM_WORDS, in_tri)

    # padding
    padded = pad_sequences([as_tokens], padding='post', maxlen=MAXLEN)
    print('padded')

    # get model
    model = create_model()
    print('model done')
    
    # make_pred
    pred = (model.predict(padded) > 0.5).astype("int32")
    return pred

if __name__ == '__main__':
    test1 = 'atgagagtgatggggatcaagaggaactgtcaacaatggtggatatggggaatcttaggcttttggatgctaatgatttgtaatggaagggagaacatgtgggtcacagtctattatggggtacctgtgtggaaagaagcaaaaactactctattttgtgcatcagatgctaaagcatatgagaaagaagtgcataatgtctgggctacacatgcctgtgtacccacagaccccaacccacaagaaatggagttaaaaaatgtaacagaaaattttaacatgtggaaaaatgacatggtggatcaaatgcacgaggatataattagtttatgggatcaaagcctaaaaccatgtgtaaagttgaccccactctgtgtcactttaaactgtagtgctaccagcaatagtagtacttacaataatgtcacctacaatgagaccacaaaaggagacatgaaaaattgctctttcaatataaccacagaagtaagggataagaaaaagaaggaatatgcacttttttataggcttgatataacacctcttgatgagaaatccaatgacagtgagtatagattaataaattgtaatacctcagccataacacaagcctgtccaaaggtcacttttgacccaattcctatacattattgtactccagctggttatgcgattctaaagtgtaataataagacattcaatggaacaggaccatgcaataacgtcagcactgtacaatgtacacatggaattaagccagtggtatcaactcaactactgttaaacggtagtctagcagaagaagggataataattagatctgaaaatataacagacaatgtcaaaacaataatagtacatcttaatgaacctgtagaaattgtgtgtcaaaggcccggcaataacacaagacaaagtgtgaggataggaccaggacaaacattctatgcaacaggagacataataggagatataagagcagcacattgtaacattactgaagagcaatggaataaaactttaaacagggtaagagaaaaattaggagaatacttccctaatagaacaataaaatttgatcaacactcaggaggggacttagaaattacaacacatagctttaattgtagaggagaatttttctattgcaatacatcaaaattgttcacatacatgtggcctaacagtacaggagatacttcaaattcaaaaaacatcacaatccgatgcagaataagacaaattataaacatgtggcagggggtaggacgagcaatgtatgcccctcctgttgaagggaacataacatgtagatcaaatatcacaggactactattgacacgtgatggaggtaatggtaatgcagaaaatggctcagaaatattcagacctgcaggaggagatatgagggacaattggagaagtgaattatataaatataaagtgatagaaattaagccattaggactggcacccactaaggcaaaaaggcgagtggtggagagagaaaaaagagcagtgggaataggagctatgttccttgggttcttgggagtagcaggaagcactatgggcgcagcatcaataacgctgacggtacaggccagacaactgttgtctggtatagtgcaacagcaaagcaatttgctgaaggctatagaggcgcaacagcatctgttgcaactcacggtctggggcattaaacagctccaggcaagagtcctggctatggaaagatacctaaaggatcaacagctcctagggatttggggctgctctggaaaacgcatctgcaccactgccgtgccttggaacgccagttggagtaataaatcttacgagagaatttgggataacatgacatggatgcagtgggatagagaaattagtaactacacagacacaatatacaggttgcttgaagactcgcaaaaccagcaggaagaaaatgaaaaggagttactagaattggacagatggaacaatctgtggaattggtttggcataacaaactggctgtggtatataaaaatattcataatgatagtaggaggcttgataggtttaagaataatttttgctgtgctttctttagtaaatagagtcaggcagggatactcacctttgtcatttcagacccttaccccaaaccagaggggactcgacaggctcggaggaatcgaagaagaaggtggagagcaagacaaagacagatccattcgattagtgagcggattcttagcacttttctgggacgatctgaggagcctgtgccttttcagctaccaccgattgagagacttcatattggtgacagcgagagtggtggaacttctgggacgcagcagtctcaggggactacagaagggatgggcagcccttaagtatctgggaggtcttgtgcagtattgggggctagagctaaaaaagagtgctactagtctgcttgataccatagcaatagcagtagctgaaggaacagataggattatagaattagtacaaagaatttgtagagctatctaccacatacctacaagaataagacagggctttgaagcagctttgcaatag'
#    pred = make_pred(test1)
#    print(pred)
    parser = ArgumentParser()
    parser.add_argument('--gen', type=str,
                        required=True, help="genome to predict")
    args = parser.parse_args()

    pred = make_pred(args.gen)
    print(pred)