# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# @Time : 2023/6/4 16:30 
# @Author : sanmaomashi
# @GitHub : https://github.com/sanmaomashi
from sklearn.feature_extraction.text import TfidfVectorizer

# 我们的文档集
documents = [
    'Hello, how are you?',
    'I am getting started with Natural Language Processing.',
    'This is an example of bag of words.',
]

# 创建 TfidfVectorizer 的实例
vectorizer = TfidfVectorizer()

# 拟合并转换文档集
X = vectorizer.fit_transform(documents)

# 输出每个文档的特征向量
print(X.toarray())

# 输出每个特征的名称
print(vectorizer.get_feature_names())
