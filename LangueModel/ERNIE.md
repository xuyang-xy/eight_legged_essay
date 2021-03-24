ERNIE
-----
01. 简介: Enhanced Representation from kNowledge IntEgration  

FAQ
---
01. ERNIE 相对于BERT的改进有哪些？
(1) 百度ERNIE采用的Masked Language Model是一种带有先验知识Mask机制, 实体级别的mask,  
如果采用BERT随机mask，则根据后缀“龙江”即可轻易预测出“黑”字。  
引入了词、实体mask之后，“黑龙江”作为一个整体被mask掉了，因此模型不得不从更长距离的依赖（“冰雪文化名城”）中学习相关性  
(2) 百度ERNIE还引入了DLM（对话语言模型）任务，通过这种方式来学习相同回复对应的query之间的语义相似性  
(3) 更加适用于中文NLP的场景