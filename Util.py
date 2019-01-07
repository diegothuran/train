import pandas as pd
from setuptools.command.saveopts import saveopts
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
import keras.metrics
from keras.layers import Dense, Embedding, LSTM, GRU, SpatialDropout1D, Activation
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.python.keras.models import save_model
import numpy as np
import pickle


def read_train_data():
    df = pd.read_csv('data/Bad Rambo - treino.tsv', header=None, sep='\t')

    sentences = df[0]
    labels = df[1]
    labels = labels.T.tolist()
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(labels)

    return sentences.T.tolist(), transfomed_label

def read_test_data():
    df = pd.read_csv('data/Bad Rambo - Teste.tsv', header=None, sep='\t')

    sentences = df[1]
    labels = df[2]
    labels = labels.T.tolist()
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(labels)

    return sentences.T.tolist(), transfomed_label


def prepare_data():
    X_train, y_train = read_train_data()
    X_test, y_test = read_test_data()

    tokenizer_obj = Tokenizer()
    total_reviews = X_test + X_train
    tokenizer_obj.fit_on_texts(total_reviews)
    max_length = max([len(s.split()) for s in total_reviews])
    vocab_size = len(tokenizer_obj.word_index) + 1
    #total_labels = np.concatenate(y_train, y_test)
    print("MAX: {}".format(max_length))
    X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
    X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

    print(vocab_size)

    EMBEDDING_DIM = 100

    print('Build model...')

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Convolution1D(32, 3, padding='same'))
    model.add(Convolution1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(SpatialDropout1D(0.4))
    # model.add(LSTM(80, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Summary of the built model...')
    print(model.summary())



    print('Train...')

    model.fit(X_train_pad, y_train, batch_size=1, epochs=15,
              validation_data=(X_test_pad, y_test), verbose=1)

    test_sample1 = "A minha cadela adorou a ração"
    test_sample2 = "A cadela da atendente é ruim"
    test_samples = [test_sample1, test_sample2]

    test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

    print(model.predict(x=test_samples_tokens_pad))

    model.save("keras.h5", overwrite=True, include_optimizer=True)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    scores = model.evaluate(X_test_pad, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    prepare_data()

