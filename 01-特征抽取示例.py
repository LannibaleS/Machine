

from sklearn.feature_extraction.text import CountVectorizer

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 归一化、标准化
# import impute.SimpleImputer from sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, imputation
from sklearn.impute import SimpleImputer
import numpy as np


vector = CountVectorizer()

res = vector.fit_transform(['life is short, I like python', 'life is too long, I dislike python'])

print(vector.get_feature_names())
print(res.toarray())
# print(res)


print('\n====== 字典特征抽取 ======')
''' 字典特征抽取 '''
from sklearn.feature_extraction import DictVectorizer

def dictvec():
    ''' 字典特征抽取 '''
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit.transform
    data = dict.fit_transform([{'city': '北京','temperature':100},
                              {'city': '上海','temperature':60},
                              {'city': '深圳','temperature':30}])

    print(data)
    print(dict.inverse_transform(data))
    print(dict.get_feature_names())
    # print(data.toarray())

    return None



print('\n====== 文本特征抽取 ======')
''' 文本特征抽取 '''
from sklearn.feature_extraction.text import CountVectorizer
def countvec():
    cv = CountVectorizer()
    data = cv.fit_transform(['人生 苦短，我 喜欢 Python', '人生漫长，不用 python'])
    print(cv.get_feature_names())
    print(data.toarray())
    return None


def cut_word():
    con1 = jieba.cut('今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。')

    con2 = jieba.cut('我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。')

    con3 = jieba.cut('如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。')

    # 转化成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转化成字符串，字符串以空格隔开
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)


    return c1,c2,c3

def hanzivec():
    c1, c2, c3 = cut_word()
    print(c1,c2,c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())
    return None


def tfidfvec():
    c1, c2, c3 = cut_word()
    print(c1,c2,c3)
    cv = TfidfVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())
    return None




def mm():
    ''' 归一化处理 '''
    print('\n====== 归一化处理 ======')
    mm = MinMaxScaler(feature_range=(2,3))

    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])

    print(data)

    return None


def standard():
    ''' 标准化缩放
    1、特点：通过对原始数据进行变换把数据变换到均值为0,方差为1范围内
    在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。

    '''
    print('\n====== 标准化缩放 ======')
    std = StandardScaler()
    data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])
    print(data)



def im():
    '''
    缺失值处理

    NaN, nan
    :return:
    '''
    im = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
    data = im.fit_transform([[1,2], [np.nan, 3], [7,np.nan]])
    print(data)


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # hanzivec()
    # tfidfvec()
    # mm()
    # standard()
    im()