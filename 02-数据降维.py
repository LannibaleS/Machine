''''''
'''
特征选择：
    1、特征选择就是单纯地从提取到的所有特征中选择部分特征作为训练集特征，
    特征在选择前和选择后可以改变值、也不改变值，但是选择后的特征维数肯
    定比选择前小，毕竟我们只选择了其中的一部分特征。
    
    2、主要方法（三大武器）：Filter(过滤式):VarianceThreshold
	     	              Embedded(嵌入式)：正则化、决策树
		                  Wrapper(包裹式)
    
    3、其他特征选择方式：神经网络
'''
from sklearn.feature_selection import VarianceThreshold

# 主成分分析
'''
PCA
本质：PCA是一种分析、简化数据集的技术
目的：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
作用：可以削减回归分析或者聚类分析中特征的数量
PCA：特征数量达到上百的时候，考虑数据的简化
数据也会改变，特征数据也会减少
'''
from sklearn.decomposition import PCA

def var():
    '''
    var方差
    特征选择 -- 删除低方差的特征
    :return:
    '''
    print('\n====== 特征选择 ======')

    var = VarianceThreshold(threshold=0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)

    return None


def pca():
    '''
    主成分分析进行特征降维
    :return:
    '''
    print('\n====== PCA ======')

    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print(data)

if __name__ == '__main__':
    # var()
    pca()