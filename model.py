""" A miny inception net for mnist """


"""This is a slightly modified example of using Hierarchical RNN (HRNN) to classify MNIST digits.

This was adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_hierarchical_rnn.py

HRNNs can learn across multiple levels of temporal hiearchy over a complex sequence.
Usually, the first recurrent layer of an HRNN encodes a sentence (e.g. of word vectors)
into a  sentence vector. The second recurrent layer then encodes a sequence of
such vectors (encoded by the first layer) into a document vector. This
document vector is considered to preserve both the word-level and
sentence-level structure of the context.

# References
    - [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)
        Encodes paragraphs and documents with HRNN.
        Results have shown that HRNN outperforms standard
        RNNs and may play some role in more sophisticated generation tasks like
        summarization or question answering.
    - [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
        Achieved state-of-the-art results on skeleton based action recognition with 3 levels
        of bidirectional HRNN combined with fully connected layers.

In the below MNIST example the first LSTM layer first encodes every
column of pixels of shape (28, 1) to a column vector of shape (128,). The second LSTM
layer encodes then these 28 column vectors of shape (28, 128) to a image vector
representing the whole image. A final Dense layer is added for prediction.

"""

import os
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

# General parameters.
batch_size = 128
num_classes = 10

def create_model(row, col, pixel):

    # Embedding dimensions.
    row_hidden = 49
    col_hidden = 49

    # 4D input.
    x = Input(shape=(row, col, pixel))

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

    # Encodes columns of encoded rows.
    encoded_columns = LSTM(col_hidden)(encoded_rows)

    # Final predictions and model.
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    
    return model


def train(run=0):
    



    # The data, shuffled and split between train and test sets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshapes data to 4D for Hierarchical RNN.
    x_train = x_train.reshape(x_train.shape[0], 7, 7, 16)
    x_test = x_test.reshape(x_test.shape[0], 7, 7, 16)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Converts class vectors to binary class matrices.
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    row, col, pixel = x_train.shape[1:]

    model = create_model(row, col, pixel)

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    # Callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    checkpoint = ModelCheckpoint('best'+str(run)+'.h5',save_best_only=True, save_weights_only=True)

    # Training
    model.fit(x_train, y_train,
            batch_size=batch_size, callbacks=[early_stopping, checkpoint],
            verbose=1, validation_data=(x_test, y_test))

def evaluate():
    # Evaluation.
        # The data, shuffled and split between train and test sets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshapes data to 4D for Hierarchical RNN.
    x_train = x_train.reshape(x_train.shape[0], 7, 7, 16)
    x_test = x_test.reshape(x_test.shape[0], 7, 7, 16)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Converts class vectors to binary class matrices.
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    row, col, pixel = x_train.shape[1:]

    model = create_model(row, col, pixel)
    files = os.listdir('.')

    for f in files:
        if '.h5' in f:
            model.load_weights(f)
            scores = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])

train()