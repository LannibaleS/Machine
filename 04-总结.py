''''''


'''
特征工程

1、特征抽取
    抽取文本中的数据
    
2、特征与处理
    归一化
        目的：避免某一个特征对结果造成很大的影响
        
    标准化：
        对于异常数据来说，标准化更好，一般用标准化
        平均值、标准差、方差
        方差的大小：反映了数据的近似度
        方差小：数据接近
        方差大：数据差距大，数据离散
    
    时间处理
    缺失值
    
3、数据降维
    目的：减少数据的特征
    
    1、特征选择
        就是单纯地从提取到的所有特征中选择部分特征作为训练集特征，
        特征在选择前和选择后可以改变值、也不改变值，但是选择后的特征维数肯
        定比选择前小，毕竟我们只选择了其中的一部分特征。
    
        主要方法（三大武器）：Filter(过滤式):VarianceThreshold
	     	              Embedded(嵌入式)：正则化、决策树
		                  Wrapper(包裹式)
		                  
   2、PCA（主成分分析）
   
   
4、监督学习1
    分类、回归；输入数据有特征值有标签，即有标准答案
    
    分类    k-近邻算法、贝叶斯分类、决策树与随机森林、逻辑回归、神经网络
    回归    线性回归、岭回归
    标注    隐马尔可夫模型     (不做要求)
    
    分类和回归的判别依据：目标值是离散的还是连续的
    离散的 -》分类
    连续的 -》回归
    
5、非监督学习
    聚类
    输入数据有特征无标签，即无标准答案
    
    聚类    k-means
    

        




'''