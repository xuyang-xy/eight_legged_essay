Batch Normalization & Layer Normalization
-----------------------------------------
01. 概述: 归一化是对数据以某个维度做0均值1方差的处理  
BatchNorm 通过对batch维度归一，LayerNorm通过对Hidden size维度归一  
将数据分布拉回到均值为0方差为1的标准正态分布(稳定分布状态), 规整数据的分布解决梯度消失/爆炸的问题
02. 动机: 深度神经网络中的Internal Covariate Shift(内部变量偏移), 使得各层输入不再是独立同分布，导致  
(1) 上层参数需要适应新的输入数据分布，学习速度下降  
(2) 反向传播时低层神经网络的梯度消失, 收敛变慢乃至无法学习
03. BatchNorm 效果：  
(1) BN层加速网络学习收敛  
(2) BN更有利于梯度下降，使得梯度不会出现过大或者过小的梯度值  
04. BatchNorm 局限：  
(1) BN对于batch_size的大小敏感的，batch_size很小的时候效果不稳定  
(2) BN对于序列化模型效果不太好， NLP任务上比较少用
05. LayerNorm 效果：  
(1) LayerNorm 能够很好的应用在序列化网络上  
(2) LayerNorm 对batch size 不敏感
06. BatchNorm 与 LayerNorm的区别  
(1) 实现逻辑上，归一化的方向不同，BN对batchsize敏感，无法在线学习(bacth_size==1)  
(2) 使用效果上，BN对CNN网络效果好， LN对RNN网络效果好，LN在NLP任务上应用比较广
07. 备注：
(1) Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015 Google  
(2) Layer Normalization 2016 University of Toronto
