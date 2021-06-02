# 新闻上的文本分类

### 数据集

搜狗实验室 搜狐新闻数据 下载地址：http://www.sogou.com/labs/resource/cs.php


### 目标

1. 实践中文短文本分类
2. 运用多种机器学习（深度学习 + 传统机器学习）方法比较短文本分类处理过程与结果差别

### 工具

深度学习：keras

传统机器学习：sklearn

word2vec: https://github.com/Embedding/Chinese-Word-Vectors

参与比较的机器学习方法
1. SVM、SVM + word2vec
2. LSTM 、 LSTM + word2vec
3. MLP（多层感知机）
4. 朴素贝叶斯
5. KNN


###实验结果

------------------------------------------------------------------------
Naive Bayes  KNN    SVM    MLP    lstm    word2vec_svm    word2vec_lstm  
0.8543 | 0.4248 | 0.8444 | 0.8620 | 0.6819 | 0.8300 | 0.8374
-------------------------------------------------------------------------
