import tensorflow as tf
from tensorflow import keras

VOCAB_SIZE = 1665
MAXLEN = 3000
EMBEDDING_DIM = 50

# Define a simple sequential model
def create_model(vocab_size, embedding_dim, maxlen):
    model = tf.keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
        keras.layers.Conv1D(128, 5, activation='relu'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model