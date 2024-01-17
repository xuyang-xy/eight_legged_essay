DeepFM
----
01. 概述  
DeepFM(WDL for CTR predict)  
02. DeepFM的模型结构
(1) DeepFM 的输入处理与embedding操作
根据模型输入端处理的不同，存在2种方式
(1.1) 针对离散特征与连续特征的输入
```python
# feature_size = index.size() + value.size()
# [男&女，周几，学生&打工，体重，工资]
# field_size = 5
# feature_size = 2 + 7 + 2 + 1 + 1 = 13
# [男，周二，学生，50，200]
# feat_index = [0, 3, 9, 11, 12]
# feat_value = [1.0, 1.0, 1.0, 50.0, 200.0]
feat_index = tf.placeholder(tf.int32,   shape=[None, None], name="feat_index")
feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")
embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feat_index)
feat_value = tf.reshape(feat_value, shape=[-1, field_size, 1])
embeddings = tf.multiply(embeddings, feat_value)
```
(1.2) 针对全离散特征的输入
```python
feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feat_index)
```
(2) FM部分的推导
```python
# first orders
y_first_order = tf.nn.embedding_lookup(weights["feature_bias"], feat_index)
y_first_order = tf.reduce_sum(y_first_order, 2)
# sum_square part
summed_features_emb = tf.reduce_sum(embeddings, 1)
summed_features_emb_square = tf.square(summed_features_emb)
# square_sum part
squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
# second order
y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)
```
(3) 输出部分的推导
```python
# pairwise loss
pos_out = inference(feature_index_pos)
neg_out = inference(feature_index_neg)
prob = tf.nn.sigmoid(pos_out - neg_out)
loss = tf.reduce_mean(tf.losses.log_loss(label, prob))
```
(4) R-drop 增强
https://arxiv.org/pdf/2106.14448.pdf
```python
# kl 散度loss
loss_main = tf.reduce_mean(cross_entropy)
loss_aux = tf.reduce_mean(twoway_kl_loss(logits, logits2))
loss = loss_main + aux_weight * loss_aux
def twoway_kl_loss(self, lg1, lg2):
    p = tf.nn.softmax(lg1)
    q = tf.nn.softmax(lg2)
    KL_loss1 = tf.reduce_sum(p * tf.log(tf.div(p, q)), axis=-1)
    KL_loss2 = tf.reduce_sum(q * tf.log(tf.div(q, p)), axis=-1)
    return 0.5 * tf.add(KL_loss1, KL_loss2)
```

FAQ
---
1. 为何在输入端选择分桶处理的方式？  
(1) 离散特征的增加和减少都很容易，易于模型的快速迭代
(2) 离散化后的特征对异常数据有很强的鲁棒性，特征离散化后，模型会更稳定
(3) 便于监控上游特征数据变化
2. 
3.
4.
5.