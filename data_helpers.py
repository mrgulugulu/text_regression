# coding=gbk
import numpy as np
import pandas as pd
import re
from jieba import lcut


def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)   #\'re 转义'
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

#data_training的文本的下标是4，tfidf值的下标是7
#my_data的文本下标是4，点击量的下标是3
def load_data_and_labels(path):#读取文本的函数，可能要换成连接mysql的函数；注意是train.py读取文本的函数
    data_x, data_x_list, data_y = [], [], []#data_x为处理前的文本,格式为一个列表中包含着装着新闻内容的列表(用于输出),data_x_list是将文本变成一个大的列表形式(用于在接下来的分词处理)
    f = pd.ExcelFile(path)
    io = pd.io.excel.ExcelFile(path)
    for i in f.sheet_names:  # 读取里面每一个sheet
        dx = pd.read_excel(io, sheet_name=i, usecols=[4])  #这里是读取第五列，如果要修改读取的列数就修改这里的数字
        dy = pd.read_excel(io, sheet_name=i, usecols=[7])
        datax = dx.values.tolist()
        datay = dy.values.tolist()
        for j in datax:
            l = str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            k = [str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')]  # 这里还需要将标点符号换掉
            data_x.append(k)
            data_x_list.append(l)

        for m in datay:
            data_y.append(m[0])

    data = []
    max_sentence_length = 0
    max_paragraph_length = 0
    for id in range(len(data_x_list)):
        paragraphs = data_x_list[id]
        sentences_split = re.split('(。|！|\!|\.|？|\?)',paragraphs)
        sentences = []
        for i in range(int(len(sentences_split) / 2)):
            sent = sentences_split[2 * i] + sentences_split[2 * i + 1]
            sentences.append(sent)
        if max_paragraph_length < len(sentences):
            max_paragraph_length = len(sentences)
        for n, sentence in enumerate(sentences):
            tokens = lcut(sentence)
            if max_sentence_length < len(tokens):
                max_sentence_length = len(tokens)
            sentence = " ".join(tokens)
            sentences[n] = sentence

        data.append([id, sentences])

    print(path)
    # print("max sentence length = {}\n".format(max_sentence_length))
    # print("max_paragraph_length = {}\n".format(max_paragraph_length))

    df = pd.DataFrame(data=data, columns=["id", "sentences"])    #创建一个二维数据表，列名为id等
    x_text = df['sentences'].tolist()    #转为列表

    return x_text, data_y


def batch_iter(data, batch_size, num_epochs, shuffle=False):     #生成一个迭代器，输入x_batch时 共7200/10 * 100 = 72000次
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1   #获得每个epoch的batch数目，结果为720
    #for epoch in range(num_epochs):     #100次
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))   #随机排列一个序列，或者数组。
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]      #一次产出batch_size数量的句子-关系对


if __name__ == "__main__":
    trainFile = 'data.xlsx'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    a, b = load_data_and_labels(trainFile)
    print(len(a))
    print(len(b))





def eval_load_data_and_labels(path):#读取文本的函数，可能要换成连接mysql的函数；注意是eval.py读取文本的函数
    data_x, data_x_list, data_y = [], [], []
    f = pd.ExcelFile(path)
    io = pd.io.excel.ExcelFile(path)
    for i in f.sheet_names:  # 读取里面每一个sheet
        dx = pd.read_excel(io, sheet_name=i, usecols=[5])  # 这里是读取第五列，如果要修改读取的列数就修改这里的数字
        dy = pd.read_excel(io, sheet_name=i, usecols=[8])
        datax = dx.values.tolist()
        datay = dy.values.tolist()
        for j in datax:
            l = str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            k = [str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')]  # 这里还需要将标点符号换掉
            data_x.append(k)
            data_x_list.append(l)
        for m in datay:
            data_y.append(m[0])


    data = []
    # lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    max_paragraph_length = 0
    for id in range(len(data_x_list)):  # 主要目标是分词，y值已经处理好
        paragraphs = data_x_list[id]  # 读取文章
        sentences_split = re.split('(。|！|\!|\.|？|\?)', paragraphs)
        sentences = []
        for i in range(int(len(sentences_split) / 2)):
            sent = sentences_split[2 * i] + sentences_split[2 * i + 1]
            sentences.append(sent)
        # sentences = nltk.sent_tokenize(paragraphs)#用正则分割句子
        if max_paragraph_length < len(sentences):
            max_paragraph_length = len(sentences)
        for n, sentence in enumerate(sentences):
            # sentence = clean_str(sentence)
            tokens = lcut(sentence)
            # tokens = nltk.word_tokenize(sentence)   #用jieba分词
            if max_sentence_length < len(tokens):
                max_sentence_length = len(tokens)
            # if len(tokens) > FLAGS.max_sentence_length:
            #    print(tokens)
            sentence = " ".join(tokens)  # 有啥区别？？？
            sentences[n] = sentence

        data.append([id, sentences])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))
    print("max_paragraph_length = {}\n".format(max_paragraph_length))

    df = pd.DataFrame(data=data, columns=["id", "sentences"])  # 创建一个二维数据表，列名为id等

    x_text = df['sentences'].tolist()  # 转为列表


    return x_text, data_x, data_y  # x_text为处理后的文本(用在模型中),data_x为处理前的文本(用于输出)

