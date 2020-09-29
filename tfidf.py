# coding=gbk
import data_helpers
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import jieba


x_text, _ = data_helpers.load_data_and_labels('data_training1.xlsx')
x_1 = sum(x_text, [])
x_2 = x_1[:]
x_3 = x_1[:]
tfidfVecorizer = TfidfVectorizer(analyzer=lambda x:x.split(' '))
tfidf_matrix = tfidfVecorizer.fit_transform(x_1)
dic = tfidfVecorizer.vocabulary_
s1, s2, s3 = 0, 0, 0

# text = '9��22�գ�2020�ꡰ�����й��������ҵ��С��ҵ���´�ҵ�����������ൺ���ȿ��������Һ��󻷾�Ԥ�����ĸ���������÷���ൺ����Ӫ���÷�չ�־ֳ������䡢�ൺ���ȹ���־ֳ�Ф��㡢�ൺ�й�ҵ����Ϣ���ָ��ֳ���ΰ���쵼���α�����ί������ѡ�ֹ���200���˳�ϯ������Ļʽ�����δ����ɹ��ҹ�ҵ����Ϣ������������ָ������ҵ����Ϣ�������簲ȫ��ҵ��չ���ģ���ҵ����Ϣ������Ϣ���ģ����ൺ����Ӫ���÷�չ�֡��ൺ���ȹ�����������죬�ൺ����С��ҵ�������ġ�������ǣ��ൺ������Ƽ��������޹�˾�а죬�ǡ������й������������ĵ�һ����������ר�����������ԡ�������С��ҵ���Ժ��󣬹�������ǿ��������̬��Ϊ���⣬����ȫ��20�����240����Ŀ���������Ǻ���װ�����졢�����²��ϡ���Ϣ�������˹����ܺʹ����ݵȺ�������������Ծ��򼽡������ǡ������ǡ�ɽ���Լ������½ʡ�ݣ����к���װ�����졢�˹����ܡ���ҵ�������������²���ռ�ȳ���80%�������������������50����Ŀ�ɹ�����������������ֳ����´��б�ʾ�������ҵ��Ϊ�ൺ����������ɫ�ͷ�չ���ƣ��Ǹ��¼�����ҵ���߳ɳ���ҵ�ľ۱��裬�����ൺ�ƶ���ҵ������������չ�¾��ã�ʵ�ָ�������չ����ս�������δ���������ΪŦ����������Ϊץ�֣�����ƶ��ൺ�д��´�ҵ��������չ����Ҫƽ̨�ͷ�����̬���ܺõ�ʵ�����Դ����ٴ��¡��Դ��������̡��Դ����ٷ�չ���ƶ��ൺ�ڡ�˫ѭ��������½���ץס�����¡����ţ���ӣ����á����¡���ƪ�����£��ºó��з�չ�������壬��δ���ĳ��о����а����Ȼ�����������������ǰ�С�Ф���ֳ����´��н��ܣ����������ൺ���ȴ���ʵʩ����������չս�ԣ��ƶ��ʱ����˲š��������г��ȸ��ഴ��Ҫ�������ȼ��ۣ����������Ե����ʵ���ҡ���������ء�ɽ����ѧ�ȡ����ֺš����л����͸�УԺ��46�ҡ�Ŀǰ���ൺ�������Ӵ�����Դ�����ڴ�̤�����ҵ��չ��������ͨ���ٰ��ൺ���ʺ���Ƽ�չ���ᡢ���´�ҵ�����ȴ������»����ַ���չ�������ƽ̨���ã����ƺ���Ƽ��ɹ�ת���͡�˫��˫����������������Ƽ�����ת��Ϊ�������ơ��ൺ������Ϊ�ൺ���𡰺����ơ�����ս���͡�˫��˫��������ͷ���������縺����Ƽ�Դͷ���µ��ش�ʹ����ȫ������ȫ������Ƽ����¸ߵء�Ϊ�ˣ��ൺ����ʼ�ռ�ְѴ��´�ҵ������Ϊ�۽������ϸ��ഴ�´�ҵ��Դ����Ҫƽ̨���������̡��������š�����������������������ƽ̨�ھ�ͷ�����������¾������Ե������г����壬Ϊ���󾭼ø�������չ�����¶��ܣ�ע���»�������ί�Ծ���������Ŀ����������߶����ۡ�9��22�������·����Ŀ���ʷ׳ʣ�9��23�գ����Ծ��򼽡������ǡ������ǵ���Ŀ��չ��·�ݡ�'
# token = jieba.lcut(text)
# text1 = ' '.join(token)
x_text1, _ = data_helpers.load_data_and_labels('./test/test_1_x2_200.xlsx')
x_text2, _ = data_helpers.load_data_and_labels('./test/test_1_x2_500.xlsx')
x_text3, _ = data_helpers.load_data_and_labels('./test/test_1_x2_1000.xlsx')

x1 = sum(x_text1, [])
x2 = sum(x_text2, [])
x3 = sum(x_text3, [])
x_1.extend(x1)
x_2.extend(x2)
x_3.extend(x3)
print(len(x_1), len(x_2), len(x_3))

for _ in range(10):
    start = time.time()
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x1: x1.split(' '))
    tfidf_matrix = tfidfVecorizer.fit_transform(x_1)
    dic = tfidfVecorizer.vocabulary_
    end = time.time()
    s1 = s1 + (end - start)
print('200����{}��'.format(s1/10))

for _ in range(10):
    start = time.time()
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x2: x2.split(' '))
    tfidf_matrix = tfidfVecorizer.fit_transform(x_2)
    dic = tfidfVecorizer.vocabulary_
    end = time.time()
    s2 = s2 + (end - start)
print('500����{}��'.format(s2/10))

for _ in range(10):
    start = time.time()
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x3: x3.split(' '))
    tfidf_matrix = tfidfVecorizer.fit_transform(x_3)
    dic = tfidfVecorizer.vocabulary_
    end = time.time()
    s3 = s3 + (end - start)
print('1000����{}��'.format(s3/10))


