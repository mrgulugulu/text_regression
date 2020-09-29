# Implementation of Hierarchical Attentional Networks for Document Classification
# http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
import time
import numpy as np
from keras import backend as K#使用抽象 Keras 后端编写新代码
from keras.models import Model, load_model
from keras.engine.topology import Layer#写自己想要的层
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed, add
from keras import layers
import tensorflow as tf
from keras import callbacks
from data import data_generate, data_shuffle
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras import initializers, regularizers, constraints

# 默认值
# maxlen = 50#每个句子最大长度
# max_sentences = 50#最大句子数
# embedding_dim = 100#嵌入层大小
# validation_split = 0.2
# embeddings_index = {}
# DROP_RATE = 0.5
# training_path = "my_data.xlsx"#测试集
# testing_path = 'my_data_testing.xlsx'#训练集

maxlen = 50#每个句子最大长度
max_sentences = 50#最大句子数
embedding_dim = 200#嵌入层大小
validation_split = 0.2
embeddings_index = {}
DROP_RATE = 0.5
training_path = "data_training1.xlsx"#测试集

testing_path_500 = './test/test_1_x2_500.xlsx'#训练集
testing_path_1000 = './test/test_1_x2_1000.xlsx'
testing_path_200 = './test/test_1_x2_200.xlsx'
testing_path = 'data_testing1.xlsx'


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,
                                2 * K.arange(self.size / 2, dtype='float32'
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):

    def __init__(self, nb_head=8, size_per_head=25, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):#初始化3个矩阵变量
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
    # def compute_output_shape(self, input_shape):
    #     return input_shape[0], input_shape[-1]


class LayerNormalization(Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

# class LayerNormalization(Layer):
#
#     def __init__(self, epsilon=1e-8, **kwargs):
#         self._epsilon = epsilon
#         super(LayerNormalization, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.beta = self.add_weight(
#             shape=(input_shape[-1],),
#             initializer='zero',
#             name='beta')
#         self.gamma = self.add_weight(
#             shape=(input_shape[-1],),
#             initializer='one',
#             name='gamma')
#         super(LayerNormalization, self).build(input_shape)
#
#     def call(self, inputs):
#
#         mean, variance = tf.nn.moments(inputs, [-1])
#         normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
#         outputs = self.gamma * normalized + self.beta
#         return outputs
#
#     def compute_output_shape(self, input_shape):
#         return input_shape




x_train, y_train, word_index = data_generate(training_path)
x_train, y_train = data_shuffle(x_train, y_train)

# x_test1, y_test1, _ = data_generate(testing_path_200)
# x_test2, y_test2, _ = data_generate(testing_path_500)
# x_test3, y_test3, _ = data_generate(testing_path_1000)
x_test, y_test, _ = data_generate(testing_path)



embedding_layer = Embedding(len(word_index) + 1, embedding_dim, embeddings_initializer=glorot_normal(1),#这里用了xavier初始化
                            input_length=maxlen, trainable=True)

#对于模块的组合，可行的有,bi-Gru+attention+globalaveragepooling1d+LN
sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
# embedded_sequences = Position_Embedding()(embedded_sequences)
# lstm_word = Bidirectional(GRU(100,  return_sequences=True, dropout=0.2))(embedded_sequences)
# attn_word = Attention(8, 25)([lstm_word, lstm_word, lstm_word])
attn_word = Attention(8, 25)([embedded_sequences, embedded_sequences, embedded_sequences])

#add&norm
# attn_word = add([lstm_word, attn_word])
attn_word = add([embedded_sequences, attn_word])
attn_word = LayerNormalization()(attn_word)
attn_word = layers.GlobalAveragePooling1D()(attn_word)
sentenceEncoder = Model(sentence_input, attn_word)

review_input = Input(shape=(max_sentences, maxlen), dtype='int32')
review_encoder = TimeDistributed(sentenceEncoder)(review_input)#要在每一个timestep上运行一次sentenceEncoder这个model,也就是对每一句进行计算
# lstm_sentence = Bidirectional(GRU(100, return_sequences=True, dropout=0.2))(review_encoder)
# attn_sentence = Attention(8, 25)([lstm_sentence, lstm_sentence, lstm_sentence])
attn_sentence = Attention(8, 25)([review_encoder, review_encoder, review_encoder])

# #add&norm
attn_sentence = add([attn_sentence, review_encoder])#残差
# attn_sentence = add([attn_sentence, lstm_sentence])#残差
attn_sentence = LayerNormalization()(attn_sentence)
# attn_sentence = layers.GlobalAveragePooling1D()(attn_sentence)
# atte_sentence = Bidirectional(GRU(100))(attn_sentence)
attn_sentence1 = Attention(8, 25)([attn_sentence, attn_sentence, attn_sentence])
attn_sentence = add([attn_sentence, attn_sentence1])
attn_sentence = LayerNormalization()(attn_sentence)

attn_sentence = layers.GlobalAveragePooling1D()(attn_sentence)
preds = Dense(1)(attn_sentence)
model = Model(review_input, preds)
model.summary()


optimizer = Adam(0.0001)
model.compile(loss='mae', optimizer=optimizer)#这里用均值误差作为metrics

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=1
    ),
    callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True
    )

]

history = model.fit(x_train, y_train, validation_split=0.1, callbacks=callbacks_list, epochs=20, batch_size=64)


model = load_model('my_model.h5', custom_objects={'Attention': Attention, 'LayerNormalization': LayerNormalization})
result1 = model.evaluate(x_test, y_test)
print('\n')
print(result1)




