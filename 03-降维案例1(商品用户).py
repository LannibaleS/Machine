import pandas as pd
from sklearn.decomposition import PCA


# 读取四张表的数据1
prior = pd.read_csv('../data/order_products__prior.csv')
products = pd.read_csv('../data/products.csv')
orders = pd.read_csv('../data/orders.csv')
aisles = pd.read_csv('../data/aisles.csv')

# 合并四张表到一张表（用户-物品类别）
_mg = pd.merge(prior, products, on=['product_id','product_id'])
_mg = pd.merge(_mg, orders, on=['order_id','order_id'])
merge_table = pd.merge(_mg, aisles, on=['aisle_id','aisle_id'])

# 查看前十行
print(merge_table.head(10))


# 交叉表（特殊的分组工具）
cross = pd.crosstab(merge_table['user_id'], merge_table['aisle'])
print(cross)

# 进行主成分分析
pca = PCA(n_components=0.9)

data = pca.fit_transform(cross)
print(data)
