# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/4 18:30 
# @Author : sanmaomashi
# @GitHub : https://github.com/sanmaomashi

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载手写数字数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 创建一个t-SNE对象，设置降维后的维度为2
tsne = TSNE(n_components=2)

# 对数据进行t-SNE降维
X_reduced = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], alpha=0.5, label=str(i))
plt.legend()
plt.title('t-SNE of Digits dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
