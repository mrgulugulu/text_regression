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
    text = re.sub(r"\'re", " are ", text)   #\'re ת��'
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

#data_training���ı����±���4��tfidfֵ���±���7
#my_data���ı��±���4����������±���3
def load_data_and_labels(path):#��ȡ�ı��ĺ���������Ҫ��������mysql�ĺ�����ע����train.py��ȡ�ı��ĺ���
    data_x, data_x_list, data_y = [], [], []#data_xΪ����ǰ���ı�,��ʽΪһ���б��а�����װ���������ݵ��б�(�������),data_x_list�ǽ��ı����һ������б���ʽ(�����ڽ������ķִʴ���)
    f = pd.ExcelFile(path)
    io = pd.io.excel.ExcelFile(path)
    for i in f.sheet_names:  # ��ȡ����ÿһ��sheet
        dx = pd.read_excel(io, sheet_name=i, usecols=[4])  #�����Ƕ�ȡ�����У����Ҫ�޸Ķ�ȡ���������޸����������
        dy = pd.read_excel(io, sheet_name=i, usecols=[7])
        datax = dx.values.tolist()
        datay = dy.values.tolist()
        for j in datax:
            l = str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            k = [str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')]  # ���ﻹ��Ҫ�������Ż���
            data_x.append(k)
            data_x_list.append(l)

        for m in datay:
            data_y.append(m[0])

    data = []
    max_sentence_length = 0
    max_paragraph_length = 0
    for id in range(len(data_x_list)):
        paragraphs = data_x_list[id]
        sentences_split = re.split('(��|��|\!|\.|��|\?)',paragraphs)
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

    df = pd.DataFrame(data=data, columns=["id", "sentences"])    #����һ����ά���ݱ�����Ϊid��
    x_text = df['sentences'].tolist()    #תΪ�б�

    return x_text, data_y


def batch_iter(data, batch_size, num_epochs, shuffle=False):     #����һ��������������x_batchʱ ��7200/10 * 100 = 72000��
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1   #���ÿ��epoch��batch��Ŀ�����Ϊ720
    #for epoch in range(num_epochs):     #100��
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))   #�������һ�����У��������顣
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]      #һ�β���batch_size�����ľ���-��ϵ��


if __name__ == "__main__":
    trainFile = 'data.xlsx'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    a, b = load_data_and_labels(trainFile)
    print(len(a))
    print(len(b))





def eval_load_data_and_labels(path):#��ȡ�ı��ĺ���������Ҫ��������mysql�ĺ�����ע����eval.py��ȡ�ı��ĺ���
    data_x, data_x_list, data_y = [], [], []
    f = pd.ExcelFile(path)
    io = pd.io.excel.ExcelFile(path)
    for i in f.sheet_names:  # ��ȡ����ÿһ��sheet
        dx = pd.read_excel(io, sheet_name=i, usecols=[5])  # �����Ƕ�ȡ�����У����Ҫ�޸Ķ�ȡ���������޸����������
        dy = pd.read_excel(io, sheet_name=i, usecols=[8])
        datax = dx.values.tolist()
        datay = dy.values.tolist()
        for j in datax:
            l = str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')
            k = [str(j[0]).strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ')]  # ���ﻹ��Ҫ�������Ż���
            data_x.append(k)
            data_x_list.append(l)
        for m in datay:
            data_y.append(m[0])


    data = []
    # lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    max_paragraph_length = 0
    for id in range(len(data_x_list)):  # ��ҪĿ���Ƿִʣ�yֵ�Ѿ������
        paragraphs = data_x_list[id]  # ��ȡ����
        sentences_split = re.split('(��|��|\!|\.|��|\?)', paragraphs)
        sentences = []
        for i in range(int(len(sentences_split) / 2)):
            sent = sentences_split[2 * i] + sentences_split[2 * i + 1]
            sentences.append(sent)
        # sentences = nltk.sent_tokenize(paragraphs)#������ָ����
        if max_paragraph_length < len(sentences):
            max_paragraph_length = len(sentences)
        for n, sentence in enumerate(sentences):
            # sentence = clean_str(sentence)
            tokens = lcut(sentence)
            # tokens = nltk.word_tokenize(sentence)   #��jieba�ִ�
            if max_sentence_length < len(tokens):
                max_sentence_length = len(tokens)
            # if len(tokens) > FLAGS.max_sentence_length:
            #    print(tokens)
            sentence = " ".join(tokens)  # ��ɶ���𣿣���
            sentences[n] = sentence

        data.append([id, sentences])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))
    print("max_paragraph_length = {}\n".format(max_paragraph_length))

    df = pd.DataFrame(data=data, columns=["id", "sentences"])  # ����һ����ά���ݱ�����Ϊid��

    x_text = df['sentences'].tolist()  # תΪ�б�


    return x_text, data_x, data_y  # x_textΪ�������ı�(����ģ����),data_xΪ����ǰ���ı�(�������)

