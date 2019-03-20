神经网络如何改善梯度消失、梯度爆炸？
合适的激活函数（ReLu之类的）、增加残差网络、批标准化、减少网络深度


BN batch normalization 批标准化
改进版：Layer Normalization 层标准化，在 Transformer （Attention is all you need ）中有提到。
好处：可以减少计算，加快训练收敛速度，跳出梯度饱和点，改善梯度消失，梯度爆炸问题。