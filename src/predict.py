import sys
sys.path.append(".")
import logging
log = logging.getLogger(__name__)

from argparse import ArgumentParser
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from nltk import ngrams
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 1665
NUM_WORDS = 5000
MAXLEN = 3000
EMBEDDING_DIM = 50

def tokenize_form_file(NUM_WORDS, ngrams):
    log.info("Getting tokenizer")

    with open('src/models_files/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenized = tokenizer.texts_to_sequences(ngrams)
    return tokenized

def recreate_model():
    log.info("Getting the model")
    model = tf.keras.models.load_model('./src/models_files/model.h5')
    return model

def make_pred(string_input):
    log.info("Start of preprocessing")
    # from string to trigrams
    in_tri = np.array([''.join(i) for i in ngrams(string_input, 3)])
 
    # tokenizing
    as_tokens = tokenize_form_file(NUM_WORDS, in_tri)
    as_tokens = [x for x in as_tokens if len(x) == 1]

    # padding
    padded = pad_sequences([as_tokens], padding='post', maxlen=MAXLEN)

    # get model
    model = recreate_model()
    
    log.info("Start of the predictions")
    # make_pred
    pred = (model.predict(padded) > 0.5).astype("int32")
    return pred

if __name__ == '__main__':
#    test1 = 'atgagagtgatggggatcaagaggaactgtcaacaatggtggatatggggaatcttaggcttttggatgctaatgatttgtaatggaagggagaacatgtgggtcacagtctattatggggtacctgtgtggaaagaagcaaaaactactctattttgtgcatcagatgctaaagcatatgagaaagaagtgcataatgtctgggctacacatgcctgtgtacccacagaccccaacccacaagaaatggagttaaaaaatgtaacagaaaattttaacatgtggaaaaatgacatggtggatcaaatgcacgaggatataattagtttatgggatcaaagcctaaaaccatgtgtaaagttgaccccactctgtgtcactttaaactgtagtgctaccagcaatagtagtacttacaataatgtcacctacaatgagaccacaaaaggagacatgaaaaattgctctttcaatataaccacagaagtaagggataagaaaaagaaggaatatgcacttttttataggcttgatataacacctcttgatgagaaatccaatgacagtgagtatagattaataaattgtaatacctcagccataacacaagcctgtccaaaggtcacttttgacccaattcctatacattattgtactccagctggttatgcgattctaaagtgtaataataagacattcaatggaacaggaccatgcaataacgtcagcactgtacaatgtacacatggaattaagccagtggtatcaactcaactactgttaaacggtagtctagcagaagaagggataataattagatctgaaaatataacagacaatgtcaaaacaataatagtacatcttaatgaacctgtagaaattgtgtgtcaaaggcccggcaataacacaagacaaagtgtgaggataggaccaggacaaacattctatgcaacaggagacataataggagatataagagcagcacattgtaacattactgaagagcaatggaataaaactttaaacagggtaagagaaaaattaggagaatacttccctaatagaacaataaaatttgatcaacactcaggaggggacttagaaattacaacacatagctttaattgtagaggagaatttttctattgcaatacatcaaaattgttcacatacatgtggcctaacagtacaggagatacttcaaattcaaaaaacatcacaatccgatgcagaataagacaaattataaacatgtggcagggggtaggacgagcaatgtatgcccctcctgttgaagggaacataacatgtagatcaaatatcacaggactactattgacacgtgatggaggtaatggtaatgcagaaaatggctcagaaatattcagacctgcaggaggagatatgagggacaattggagaagtgaattatataaatataaagtgatagaaattaagccattaggactggcacccactaaggcaaaaaggcgagtggtggagagagaaaaaagagcagtgggaataggagctatgttccttgggttcttgggagtagcaggaagcactatgggcgcagcatcaataacgctgacggtacaggccagacaactgttgtctggtatagtgcaacagcaaagcaatttgctgaaggctatagaggcgcaacagcatctgttgcaactcacggtctggggcattaaacagctccaggcaagagtcctggctatggaaagatacctaaaggatcaacagctcctagggatttggggctgctctggaaaacgcatctgcaccactgccgtgccttggaacgccagttggagtaataaatcttacgagagaatttgggataacatgacatggatgcagtgggatagagaaattagtaactacacagacacaatatacaggttgcttgaagactcgcaaaaccagcaggaagaaaatgaaaaggagttactagaattggacagatggaacaatctgtggaattggtttggcataacaaactggctgtggtatataaaaatattcataatgatagtaggaggcttgataggtttaagaataatttttgctgtgctttctttagtaaatagagtcaggcagggatactcacctttgtcatttcagacccttaccccaaaccagaggggactcgacaggctcggaggaatcgaagaagaaggtggagagcaagacaaagacagatccattcgattagtgagcggattcttagcacttttctgggacgatctgaggagcctgtgccttttcagctaccaccgattgagagacttcatattggtgacagcgagagtggtggaacttctgggacgcagcagtctcaggggactacagaagggatgggcagcccttaagtatctgggaggtcttgtgcagtattgggggctagagctaaaaaagagtgctactagtctgcttgataccatagcaatagcagtagctgaaggaacagataggattatagaattagtacaaagaatttgtagagctatctaccacatacctacaagaataagacagggctttgaagcagctttgcaatag'
#    pred = make_pred(test1)
#    print(pred)
    parser = ArgumentParser()
    parser.add_argument('--gen', type=str,
                        required=True, help="genome to predict")
    args = parser.parse_args()

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    
    logging.basicConfig(filename=f'logs/{dt_string}_predict.log', filemode='w',
                        level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    log.info("Starting predictions process")
    pred = make_pred(args.gen)
    log.info(f"Finished, predicted {pred}")