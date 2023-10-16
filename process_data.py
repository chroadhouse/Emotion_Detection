from tensorflow import keras 
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json

def get_glove_matrix(vocab_size, tokenizer, embedding_dim):
    glove_path = 'GloVe/glove.6b/glove.6b.200d.txt'
    num_tokens = vocab_size
    #embedding_dim = 200 #latent factors or features  
    hits = 0
    misses = 0
    embeddings_index = {}

    # Read word vectors
    with open(glove_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    # Now assign word vectors for these 
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

def process_text_intensifer(training_done, data):
    texts = data['content'].tolist()
    label = data['intensity'].tolist()

    if training_done:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('Models/label_encoder_intensifer.npy',allow_pickle=True)
        
        tokenizer = None
        with open('Models/tokenizer_file_intensifer.json', 'r') as json_file:
            json_string = json_file.read()
            tokenizer = tokenizer_from_json(json_string)
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        label_encoder = LabelEncoder()
        encoded_label = label_encoder.fit_transform(label)

    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = 229
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
    num_classes = len(set(label_encoder.classes_))
    
    encoded_categorical_label = to_categorical(encoded_label, num_classes)

    vocab_size = len(tokenizer.word_index)+1
    embedding_dim = 200

    embedding_matrix = get_glove_matrix(vocab_size, tokenizer, embedding_dim)

    if not training_done:

        # Save the label encoder here
        np.save('Models/label_encoder_intensifer.npy', label_encoder.classes_)

        tokenizer_json = tokenizer.to_json()
        with open('Models/tokenizer_file_intensifer.json', 'w') as json_file:
            json_file.write(tokenizer_json)

    return padded_sequence, encoded_categorical_label, embedding_matrix, vocab_size, max_sequence_length, num_classes, embedding_dim

        

def process_text_TBED(training_done, data):
    print(data['sentiment'].value_counts())
    texts = data['content'].tolist()
    label = data['sentiment'].tolist()
        
    if training_done:

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('Models/label_encoder_TBED.npy',allow_pickle=True)
        

        tokenizer = None
        with open('Models/tokenizer_file_TBED.json', 'r') as json_file:
            json_string = json_file.read()
            tokenizer = tokenizer_from_json(json_string)
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

        label_encoder = LabelEncoder()
        encoded_label = label_encoder.fit_transform(label)

        
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = 229
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding="pre")
    
    encoded_label = label_encoder.fit_transform(label)
    num_classes = len(set(label_encoder.classes_))
    print(num_classes)
    encoded_categorical_label = to_categorical(encoded_label, num_classes)

    vocab_size = len(tokenizer.word_index)+1
    embedding_dim = 200

    embedding_matrix = get_glove_matrix(vocab_size, tokenizer, embedding_dim)

    if not training_done:

        # Save the label encoder here
        np.save('Models/label_encoder_TBED.npy', label_encoder.classes_)

        tokenizer_json = tokenizer.to_json()
        with open('Models/tokenizer_file_TBED.json', 'w') as json_file:
            json_file.write(tokenizer_json)

    return padded_sequence, encoded_categorical_label, embedding_matrix, vocab_size, max_sequence_length, num_classes, embedding_dim  


