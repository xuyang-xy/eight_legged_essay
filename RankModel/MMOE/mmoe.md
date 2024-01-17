MMOE
----
01. 概述  
MMOE(Multi-gate Mixture-of-Experts)  
02. MMOE的结构推导
```python
embedding = embedding_layer(feat_index)
task_weight = gate_layer(embedding)
task_input = tf.reduce_sum(tf.multiply(gate_weight, expert_stack), -1)
task_output = task_tower(task_input)
```



FAQ
---
1. MMOE的极化是什么以及解决方式
极化：少数expert的权重接近1，其它expert权重接近0
解决：对gate使用dropout；增加专家数；对输入gate输入多加一层变换
2. 如何平衡多任务学习间的loss
(1) 确定多任务的主次，把权重当超参去调整
(2) 把共享权重加入可学习参数，大loss的task给予小权重，小loss的task给予大权重
(3) 梯度正则