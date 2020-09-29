import os
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras import backend as K#使用抽象 Keras 后端编写新代码
from keras.models import Model, load_model
from keras import initializers
from keras.engine.topology import Layer#写自己想要的层
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed, add
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from data import data_generate, data_shuffle

maxlen = 50
max_sentences = 50
max_words = 20000
embedding_dim = 100
validation_split = 0.2
reviews = []
labels = []
texts = []
glove_dir = "./glove.6B"
embeddings_index = {}
training_path = "data_training1.xlsx"
testing_path = 'data_testing1.xlsx'


# class defining the custom attention layer
class HierarchicalAttentionNetwork(Layer):

    def __init__(self, attention_dim=100):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):#这里要接一个二维的
        return input_shape[0], input_shape[-1]


# def remove_html(str_a):
#     p = re.compile(r'<.*?>')
#     return p.sub('', str_a)
#
#
# # replace all non-ASCII (\x00-\x7F) characters with a space
# def replace_non_ascii(str_a):
#     return re.sub(r'[^\x00-\x7f]', r'', str_a)
#
#
# # Tokenization/string cleaning for dataset
# def clean_str(string):
#     string = re.sub(r"\\", "", string)
#     string = re.sub(r"\'", "", string)
#     string = re.sub(r"\"", "", string)
#     return string.strip().lower()
#
#
# input_data = pd.read_csv('labeledTrainData.tsv', sep='\t')
#
# for idx in range(input_data.review.shape[0]):
#     text = BeautifulSoup(input_data.review[idx], features="html5lib")
#     text = clean_str(text.get_text())
#     texts.append(text)
#     sentences = tokenize.sent_tokenize(text)
#     reviews.append(sentences)
#     labels.append(input_data.sentiment[idx])
#
# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(texts)
#
# data = np.zeros((len(texts), max_sentences, maxlen), dtype='int32')
#
# for i, sentences in enumerate(reviews):
#     for j, sent in enumerate(sentences):
#         if j < max_sentences:
#             wordTokens = text_to_word_sequence(sent)
#             k = 0
#             for _, word in enumerate(wordTokens):
#                 if k < maxlen and tokenizer.word_index[word] < max_words:
#                     data[i, j, k] = tokenizer.word_index[word]
#                     k = k + 1

# word_index = tokenizer.word_index
# print('Total %s unique tokens.' % len(word_index))
#
# labels = to_categorical(np.asarray(labels))
# print('Shape of reviews (data) tensor:', data.shape)
# print('Shape of sentiment (label) tensor:', labels.shape)
#
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(validation_split * data.shape[0])
#
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]

# print('Number of positive and negative reviews in training and validation set')
# print(y_train.sum(axis=0))
# print(y_val.sum(axis=0))
#

# f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

# print('Total %s word vectors.' % len(embeddings_index))
x_train, y_train, word_index = data_generate(training_path)
x_train, y_train = data_shuffle(x_train, y_train)
#
x_test, y_test, _ = data_generate(testing_path)


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=True, mask_zero=True)

sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
lstm_word = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
attn_word = HierarchicalAttentionNetwork(100)(lstm_word)
sentenceEncoder = Model(sentence_input, attn_word)

review_input = Input(shape=(max_sentences, maxlen), dtype='int32')
review_encoder = TimeDistributed(sentenceEncoder)(review_input)
lstm_sentence = Bidirectional(GRU(50, return_sequences=True))(review_encoder)
attn_sentence = HierarchicalAttentionNetwork(100)(lstm_sentence)
preds = Dense(1)(attn_sentence)
model = Model(review_input, preds)

model.compile(loss='mae', optimizer='adam')

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=1
    ),
    callbacks.ModelCheckpoint(
        filepath='my_model_han.h5',
        monitor='val_loss',
        save_best_only=True
    )

]

history = model.fit(x_train, y_train, callbacks=callbacks_list, validation_split=0.1,  epochs=10, batch_size=64)
result = model.evaluate(x_test, y_test)
print('\n', result)


# model = load_model('my_model_han.h5', custom_objects={'HierarchicalAttentionNetwork': HierarchicalAttentionNetwork})
# result = model.evaluate(x_val, y_val)
# print('/n')
# print(result[1])
# history_dict = history.history
# acc = history_dict['acc']
# val_acc_values = history_dict['val_acc']
#
# epochs = range(1, len(acc)+1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('training and validation acc')
# plt.xlabel('epochs')
# plt.ylabel('acc')
# plt.legend()
# plt.show()
