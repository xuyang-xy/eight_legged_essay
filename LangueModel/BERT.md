BERT
----
01. 概述  
BERT(Bidirectional Encoder Representations from Transformers)
02. embedding  
(1) token embedding 词嵌入  
(2) segment embedding 段嵌入  
(3) position embedding 位置嵌入  
03. multi-head-attention 多头注意力表示
(1) self attention 自注意力机制  
(2) multi-head 多头机制   
04. 前馈网络与层归一化  
feed-forward && layernorm  
05. 残差连接  

FAQ
---
01. BERT中的position emdedding方式？  
使用随机初始化的嵌入方法，pytorch官方实现中  
```
self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
```
02. self-attention机制的意义？  
用文本中的其它词来增强目标词的语义表示，从而更好的利用上下文的信息