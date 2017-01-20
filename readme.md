[TOC]

#数据分析
1. 去除TE=0的值
![TE](http://ww4.sinaimg.cn/large/006y8lVajw1fbkbgj08xxj31hg0yy7f0.jpg)
41165行，无空值。

2. NPP：
![NPP](http://ww3.sinaimg.cn/large/006y8lVajw1fbkazm2x9wj31ik0yck7x.jpg)
无空值，中值500左右，基本符合正态分布。
和TE的关系：
![NPP&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_1.png)

3. Light：
![](http://ww3.sinaimg.cn/large/006y8lVajw1fbkbtm2w4xj31hu0ych2i.jpg)
和TE关系：
![Light&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_2.png)

4. SO2：
![](http://ww1.sinaimg.cn/large/006y8lVajw1fbkbmcd065j31i60y8wqv.jpg)
和TE关系：
![SO2&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_3.png)

5. NO2：
![](http://ww4.sinaimg.cn/large/006y8lVajw1fbkbnms7vij31jg0yq7ii.jpg)
和TE关系：
![NO2&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_4.png)

6. LSTV：
![](http://ww1.sinaimg.cn/large/006y8lVajw1fbkboz8omvj31ke0yinbl.jpg)
和TE关系：	
![Light&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_5.png)

7.数据分析
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_12.png)
TE列，可以看到最后一个四分位数的差异和前面的差了一个数量级。
NPP列，总体来看，差异可以接受。
Light，最后一个四分位数的差异也很大。
SO2,总体来看，差异可以接受,但明显第一个四分位数有大量0值。
NO2、LSTV的差异也是一样，可以接受。
 对于差异相对较大的数据，这样的跨度可能是两种情况，一是部分数据是异常点，二是这些数据的分布在尾部变得稀疏，需要进一步考虑。如果是异常点，将在此数据的基础上训练模型，判断模型预测错误的情况，是否与这些异常有关。如果缺失是这样，可以采取步骤进行矫正。比如，可以复制这些预测模型表现不好的例子，以提高其在训练集中的比例；或者，把不好的例子分离出来，然后单独训练；当然也可以把这些例子去除。

#线性回归
1. 数据相关性：
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_6.png)
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/cor.png)
2. 使用有相对显著相关性的Light计算结果：
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_7.png)
3. +SO2,NO2
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_8.png)
4. 所有X
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_9.png)

#贝叶斯
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_10.png)

#多项式
5次
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_11.png)

#其他
取log，效果很差

----












