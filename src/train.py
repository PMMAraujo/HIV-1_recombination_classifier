import os

import sys
sys.path.append(".")

import logging
log = logging.getLogger(__name__)

from src import preprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras

import pandas as pd


INPUT = 'data/genomes_nonB.csv'
#VOCAB_SIZE = 1665
MAXLEN = 3000
EMBEDDING_DIM = 50
TEST_SIZE = 0.25
RANDOM_STATE = 100
NUM_WORDS = 5000



def prep_data(input=INPUT, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                num_words=NUM_WORDS, maxlen=MAXLEN, is_df=False):
    log.info("Start of the preprocessing step")

    sentences, y = preprocess.create_trigrams(input, is_df)
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences,
                                                y, test_size=test_size,
                                                random_state=random_state,
                                                stratify=y)

    X_train, X_test, vocab_size = preprocess.trigrams_tokenizer(sentences_train,
                                                            sentences_test,
                                                            num_words=num_words)

    X_train, X_test = preprocess.padding(X_train, X_test, maxlen=maxlen)

    return X_train, X_test, y_train, y_test, vocab_size

def create_model(vocab_size, embedding_dim=EMBEDDING_DIM, maxlen=MAXLEN):
    log.info("Creating model")

    METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='auc')]

    model = tf.keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        keras.layers.Conv1D(128, 5, activation='relu'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=METRICS)

    return model


if __name__ == '__main__':

    logging.basicConfig(filename='train.log', filemode='w', level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    log.info("Reading data")
    df = pd.read_csv(INPUT)
    df = df.sample(frac=0.1, random_state=1)

    log.info("Train-test split")
    X_train, X_test, y_train, y_test, vocab_size = prep_data(input=df,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, num_words=NUM_WORDS,
            maxlen=MAXLEN, is_df=True)


#    X_train, X_test, y_train, y_test, vocab_size = prep_data(input=INPUT,
#            test_size=TEST_SIZE, random_state=RANDOM_STATE, num_words=NUM_WORDS,
#            maxlen=MAXLEN)
    
    model = create_model(vocab_size, embedding_dim=EMBEDDING_DIM, maxlen=MAXLEN)

    early_stopping = tf.keras.callbacks.EarlyStopping(
                                                    monitor='val_auc', 
                                                    verbose=1,
                                                    patience=5,
                                                    mode='max',
                                                    restore_best_weights=True)
    
    log.info("Model training")
                                                    
    history = model.fit(X_train, y_train,
                    epochs=5,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    callbacks = [early_stopping],
                    batch_size=256)

    log.info("Making predictions")
    y_pred = model.predict_classes(X_test)

    log.info("Saving files")
    log.info(classification_report(y_test, y_pred))
    
    hist_df = pd.DataFrame(history.history) 
    # or save to csv: 
    hist_csv_file = 'src/models_files/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model.save("src/models_files/model.h5")
