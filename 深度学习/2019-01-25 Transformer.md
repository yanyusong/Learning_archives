## Transformer 、self-attention 特点：
可并行，速度快
可解决长距离依赖，
在符号表示序列长度比表示纬度小的时候，比RNN快，在长序列上，可通过限制只考虑该 input词中心周边 r 范围内的词。  
self-attention + point-wise 乘，论文上说效果等同于卷积，但同时又很大改善了长距离依赖问题。所以它同时集成了 CNN 可并行速度快，RNN 可长距离依赖的优点，并且效果更好。
添加了 position-embedding 来记录序列相对和绝对位置，论文中用了 sin和cos 函数，然后竟然说假设其可以学习到序列的相对位置。。。
Masked 的作用是解码时盖住后边的，不然就没法预测了
## 模型结构

![](./_image/2019-01-29-10-41-40.jpg?r=54)
![](./_image/2019-01-29-10-40-58.jpg?r=79)



