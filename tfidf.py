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

# text = '9月22日，2020年“创客中国”海洋产业中小企业创新创业大赛决赛在青岛蓝谷开赛。国家海洋环境预报中心副主任刘桂梅、青岛市民营经济发展局局长高善武、青岛蓝谷管理局局长肖焰恒、青岛市工业和信息化局副局长冯伟等领导、嘉宾、评委、参赛选手共计200余人出席决赛开幕式。本次大赛由国家工业和信息化部、财政部指导，工业和信息化部网络安全产业发展中心（工业和信息化部信息中心）、青岛市民营经济发展局、青岛蓝谷管理局联合主办，青岛市中小企业服务中心、清科启智（青岛）海洋科技创新有限公司承办，是“创客中国”开办以来的第一个海洋领域专题赛。大赛以“助力中小企业经略海洋，共筑海洋强国创新生态”为主题，吸引全国20余地市240个项目参赛，涵盖海洋装备制造、海洋新材料、信息技术、人工智能和大数据等海洋及相关领域，来自京津冀、长三角、珠三角、山东以及多个内陆省份，其中海洋装备制造、人工智能、工业互联网、海洋新材料占比超过80%。经网络初赛角逐，最终50个项目成功晋级决赛。高善武局长在致辞中表示，海洋产业作为青岛最鲜明的特色和发展优势，是高新技术企业、高成长企业的聚宝盆，更是青岛推动产业升级，大力发展新经济，实现高质量发展的主战场。本次大赛以赛事为纽带，以政策为抓手，搭建起推动青岛市创新创业高质量发展的重要平台和服务生态，很好地实现了以大赛促创新、以大赛促招商、以大赛促发展，推动青岛在“双循环”格局下紧紧抓住“创新”这个牛鼻子，做好“创新”这篇大文章，下好城市发展的先手棋，在未来的城市竞争中把握先机、掌握主动、走在前列。肖焰恒局长在致辞中介绍，近年来，青岛蓝谷大力实施创新驱动发展战略，推动资本、人才、技术、市场等各类创新要素在蓝谷集聚，引进海洋试点国家实验室、国家深海基地、山东大学等“国字号”科研机构和高校院所46家。目前，青岛蓝谷正从创新资源集聚期大踏步向产业发展期迈进。通过举办青岛国际海洋科技展览会、创新创业大赛等大型赛事活动，充分发挥展会大赛的平台作用，助推海洋科技成果转化和“双招双引”工作，将海洋科技优势转化为经济优势。青岛蓝谷作为青岛发起“海洋攻势”的主战场和“双招双引”的排头兵，积极肩负起海洋科技源头创新的重大使命，全力打造全国海洋科技创新高地。为此，青岛蓝谷始终坚持把创新创业大赛作为聚焦和整合各类创新创业资源的重要平台，以赛招商、以赛引才、以赛助创，积极借助大赛平台挖掘和孵化更多具有新经济属性的新兴市场主体，为海洋经济高质量发展培育新动能，注入新活力。评委对决赛参赛项目的质量给予高度评价。9月22日下午的路演项目精彩纷呈，9月23日，来自京津冀、长三角、珠三角的项目将展开路演。'
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
print('200的是{}秒'.format(s1/10))

for _ in range(10):
    start = time.time()
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x2: x2.split(' '))
    tfidf_matrix = tfidfVecorizer.fit_transform(x_2)
    dic = tfidfVecorizer.vocabulary_
    end = time.time()
    s2 = s2 + (end - start)
print('500的是{}秒'.format(s2/10))

for _ in range(10):
    start = time.time()
    tfidfVecorizer = TfidfVectorizer(analyzer=lambda x3: x3.split(' '))
    tfidf_matrix = tfidfVecorizer.fit_transform(x_3)
    dic = tfidfVecorizer.vocabulary_
    end = time.time()
    s3 = s3 + (end - start)
print('1000的是{}秒'.format(s3/10))


