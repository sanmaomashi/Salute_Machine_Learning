# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/4 18:29 
# @Author : sanmaomashi
# @GitHub : https://github.com/sanmaomashi

from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 创建一个PCA对象，设置保留的主成分数量为2
pca = PCA(n_components=2)

# 对数据进行PCA降维
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], color='red', alpha=0.5,label='Iris-setosa')
plt.scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], color='blue', alpha=0.5,label='Iris-versicolor')
plt.scatter(X_reduced[y == 2, 0], X_reduced[y == 2, 1], color='green', alpha=0.5,label='Iris-virginica')
plt.legend()
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()