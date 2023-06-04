# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/4 16:28 
# @Author : sanmaomashi
# @GitHub : https://github.com/sanmaomashi

# 步骤 1: 导入所需的库
from sklearn.feature_extraction.text import CountVectorizer

# 步骤 2: 创建文档数据集
documents = [
    'Hello, how are you?',
    'I am getting started with Natural Language Processing.',
    'This is an example of bag of words.',
]

# 步骤 3: 创建 CountVectorizer 的实例
vectorizer = CountVectorizer()

# 步骤 4: 拟合并转化文档数据集
X = vectorizer.fit_transform(documents)

# 查看特征向量
print(X.toarray())

# 查看每个特征对应的单词
print(vectorizer.get_feature_names())