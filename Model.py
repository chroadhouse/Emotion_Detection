import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import initializers
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.optimizers import Adam
import json
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import tokenizer_from_json, Tokenizer
from sklearn.preprocessing import LabelEncoder
import pickle

def build_model_TBED(vocab_size, embedding_dim, max_seq_length, num_classes, embedded_matrix):
    adam = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_length, weights=[embedded_matrix], trainable=False))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()
    return model


def train_model_TBED(vocab_size, embedding_dim, max_seq_length, num_classes, embeding_matrix, sequences, labels):
    model = build_model_TBED(vocab_size, embedding_dim, max_seq_length, num_classes, embeding_matrix)
    model.fit(sequences, labels, epochs=10, batch_size=256, verbose=1)
    # Model is trained now save the model
    model.save('Models/TBED_model.h5')
    # save the model 
    return model

def get_model_TBED():
    return keras.models.load_model('Models/TBED_model.h5')
    # Read in the model and return the model 

def test_model_TBED(model, sequences, labels):
    print(model.evaluate(sequences, labels))


def run_model_TBED(model, setence,max_seq_length):
    # import the tokenizer 
    tokenizer = None
    with open('Models/tokenizer_file_TBED.json', 'r') as json_file:
        json_string = json_file.read()
        tokenizer = tokenizer_from_json(json_string)

    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('Models/label_encoder_TBED.npy',allow_pickle=True)

   # label_encoder = np.load('Models/label_encoder_TBED.npy')

    sequence = tokenizer.texts_to_sequences(setence)
    padded_sequence = pad_sequences(sequence, max_seq_length, truncating='pre')
    prediction = model.predict(padded_sequence)

    # read in label encoder 

    result = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]
    probablity =  np.max(prediction)

    print(f'Emotion {result} - Probability {probablity}')
    
def run_model_intensity(model, sentence, max_seq_length):
    tokenizer = None
    with open('Models/tokenizer_file_intensifer.json', 'r') as json_file:
        json_string = json_file.read()
        tokenizer = tokenizer_from_json(json_string)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('Models/label_encoder_intensifer.npy',allow_pickle=True)

    sequence = tokenizer.texts_to_sequences(sentence)
    padded_sequence = pad_sequences(sequence, max_seq_length, truncating='pre')
    prediciton = model.predict(padded_sequence)

    result = label_encoder.inverse_transform(np.argmax(prediciton, axis=-1))[0]
    probability = np.max(prediciton)

    print(f'Intensity {result} - Probability {probability}')
        


def build_model_intensity(vocab_size, embedding_dim, max_seq_length, num_classes, embedded_matrix):
    adam = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_length, weights=[embedded_matrix], trainable=False))
    model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()
    return model


def train_model_intensity(vocab_size, embedding_dim, max_seq_length, num_classes, embedding_matrix, sequences, labels):
    model = build_model_intensity(vocab_size, embedding_dim, max_seq_length, num_classes, embedding_matrix)
    model.fit(sequences, labels, epochs=10, batch_size=256)

    model.save('Models/Intensifer_model.h5')
    return model

def get_model_intensity():
    return keras.models.load_model('Models/Intensifer_model.h5')

def run_rule_based(sentence):
    intensifiers = {
        'very':3,
        'extremely':3,
        'really':2,
        'incredibly':3,
        'so':2
    }

    intenisty_score = 1
    for word in sentence:
        if word in list(intensifiers.keys()):
            score = intensifiers[word]
            if score > intenisty_score:
                intenisty_score = score

    print(f'Rule Based System Results')
    print(f'Sentence: {sentence}')
    print(f'Intensity Score: {intenisty_score}')