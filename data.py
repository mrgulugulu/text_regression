import data_helpers
import numpy as np
import tensorflow as tf
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def data_generate(path):
    x_text, y = data_helpers.load_data_and_labels(path)  # y应该是一个batchsize*1的矩阵，也就是tfidf的矩阵
    x_text = np.array(x_text)

    # shuffle_indices = np.random.permutation(np.arange(len(y)))  # len(矩阵)返回行数
    # x_text = x_text[shuffle_indices]    #打乱顺序

    # y = np.array(y).reshape((len(y), 1))
    # y_shuffled = y[shuffle_indices]

    x_text = list(x_text)
    temp = sum(x_text, [])  # 列表降维
    # temp = list(np.array(x_text).reshape(-1))


    # dev_sample_index = -1 * int(0.1 * float(len(y)))  # -1使得x_train截至倒数1/10处，即长度为0.9
    # x_text_train_list, x_text_dev_list = x_text[:dev_sample_index], x_text[dev_sample_index:]  # 将训练集分割为9：1
    # y_train1, y_dev1 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]  # y_shuffled要reshape


    tokenizer = Tokenizer(50)
    tokenizer.fit_on_texts(temp)
    word_index = tokenizer.word_index


    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(50)
    _ = np.array(list(vocab_processor.fit_transform(temp)))  # 建立所有文档字的id映射表

    # 输入文本，建立词汇id映射表，将句子单词id化，与上句一起用
    count = -1
    x_train = []
    x_dev = []
    i = 1

    for paragraph in x_text:
        x_document = np.array(list(vocab_processor.fit_transform(paragraph)))
        x = np.array(x_document)
        x_len = len(x)
        i += 1
        if x_len < 50:
            # for i in range(FLAGS.max_paragraph_length - x_len):
            try:
                x = np.pad(x, ((0, 50 - x_len), (0, 0)))
            except ValueError:
                print('something wrong in training_data NO.{}'.format(i))
                pass
        else:
            x = x[:50]  # 这里要与"max_paragraph_length"一起改

        x_train.append(x)
    x_output = np.array(x_train)
    y_output = np.array(y)

    # for paragraph in x_text_dev_list:
    #     x_document = np.array(list(vocab_processor.fit_transform(paragraph)))
    #     x = np.array(x_document)
    #     x_len = len(x)
    #     i += 1
    #     if x_len < 50:
    #         # for i in range(FLAGS.max_paragraph_length - x_len):
    #         try:
    #             x = np.pad(x, ((0, 50 - x_len), (0, 0)))
    #         except ValueError:
    #             print('something wrong in dev_data NO.{}'.format(i))
    #             pass
    #     else:
    #         x = x[:50]
    #
    #     x_dev.append(x)
    # x_dev = np.array(x_dev)
    # y_dev = np.array(y_dev1)
    return x_output, y_output, word_index


if __name__ == "__main__":
    a,b,c= data_generate("my_data.xlsx")

def data_shuffle(x, y):#打乱顺序及分割
    np.random.seed(12)
    shuffle_indices = np.random.permutation(np.arange(len(y)))  # len(矩阵)返回行数
    x = x[shuffle_indices]    #打乱顺序

    y = np.array(y).reshape((len(y), 1))
    y_shuffled = y[shuffle_indices]

    # x_text = list(x)

    # dev_sample_index = -1 * int(0.1 * float(len(y)))  # -1使得x_train截至倒数1/10处，即长度为0.9
    # x_text_train_list, x_text_dev_list = x[:dev_sample_index], x[dev_sample_index:]  # 将训练集分割为9：1
    # y_train1, y_dev1 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]  # y_shuffled要reshape

    # return x_text_train_list, x_text_dev_list, y_train1, y_dev1
    return x, y_shuffled





# def data_generate():
#     x_text, y = data_helpers.load_data_and_labels("data_training.xlsx")  # y应该是一个batchsize*1的矩阵，也就是tfidf的矩阵
#     x_text = np.array(x_text)
#     np.random.seed(12)
#     shuffle_indices = np.random.permutation(np.arange(len(y)))  # len(矩阵)返回行数
#     x_text = x_text[shuffle_indices]    #打乱顺序
#
#     y = np.array(y).reshape((len(y), 1))
#     y_shuffled = y[shuffle_indices]
#
#     x_text = list(x_text)
#     temp = sum(x_text, [])  # 列表降维
#     # temp = list(np.array(x_text).reshape(-1))
#
#     dev_sample_index = -1 * int(0.1 * float(len(y)))  # -1使得x_train截至倒数1/10处，即长度为0.9
#     x_text_train_list, x_text_dev_list = x_text[:dev_sample_index], x_text[dev_sample_index:]  # 将训练集分割为9：1
#     y_train1, y_dev1 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]  # y_shuffled要reshape
#
#
#
#     vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(40)
#     _ = np.array(list(vocab_processor.fit_transform(temp)))  # 建立所有文档字的id映射表
#
#     # 输入文本，建立词汇id映射表，将句子单词id化，与上句一起用
#     count = -1
#     x_train = []
#     x_dev = []
#     i = 1
#
#     for paragraph in x_text_train_list:
#         x_document = np.array(list(vocab_processor.fit_transform(paragraph)))
#         x = np.array(x_document)
#         x_len = len(x)
#         i += 1
#         if x_len < 50:
#             # for i in range(FLAGS.max_paragraph_length - x_len):
#             try:
#                 x = np.pad(x, ((0, 50 - x_len), (0, 0)))
#             except ValueError:
#                 print('something wrong in training_data NO.{}'.format(i))
#                 pass
#         else:
#             x = x[:50]  # 这里要与"max_paragraph_length"一起改
#
#         x_train.append(x)
#     x_train = np.array(x_train)
#     y_train = np.array(y_train1)
#     batches = zip(x_train, y_train)
#
#     for paragraph in x_text_dev_list:
#         x_document = np.array(list(vocab_processor.fit_transform(paragraph)))
#         x = np.array(x_document)
#         x_len = len(x)
#         i += 1
#         if x_len < 50:
#             # for i in range(FLAGS.max_paragraph_length - x_len):
#             try:
#                 x = np.pad(x, ((0, 50 - x_len), (0, 0)))
#             except ValueError:
#                 print('something wrong in dev_data NO.{}'.format(i))
#                 pass
#         else:
#             x = x[:50]
#
#         x_dev.append(x)
#     x_dev = np.array(x_dev)
#     y_dev = np.array(y_dev1)
#     return x_train, x_dev, y_train, y_dev

