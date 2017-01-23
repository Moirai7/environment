[TOC]
 
 

# 数据分析
前面对每个属性的分析可以略过，这个是我最开始分析数据看的。可以从数据分析结论看起。
## NPP：
 ![NPP](http://ww3.sinaimg.cn/large/006y8lVajw1fbkazm2x9wj31ik0yck7x.jpg)      
 前面两个图，分别是boxplot和violinplot，具体看法如下图：    
 ![](http://wiki.mbalib.com/w/images/9/97/箱线图图示.jpg)   
 第三个图是将数据和正态分布比较，看数据是否匹配，这个图和最后一个直方图可以结合来看。    
 第四个图就是显示了下数据，可以看看有没有明显的outliers。  
 第五个图不用看了，看直方图就可以了。   
 可以看到，中值500左右，基本符合正态分布，0值可能影响训练数据的准确性。和TE的关系，横轴是NPP，纵轴是TE，上面是NPP的直方图，右边是TE的直方图：  
 ![NPP&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_1.png)      
 数据没有明显的分布关系。

## Light：
 ![](http://ww3.sinaimg.cn/large/006y8lVajw1fbkbtm2w4xj31hu0ych2i.jpg)      
 数据的尾部跨度很大，没有明显的规律。和TE关系：      
 ![Light&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_2.png)      

## SO2：
 ![](http://ww1.sinaimg.cn/large/006y8lVajw1fbkbmcd065j31i60y8wqv.jpg)      
 0值过多，和TE关系：      
 ![SO2&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_3.png)      

## NO2：
 ![](http://ww4.sinaimg.cn/large/006y8lVajw1fbkbnms7vij31jg0yq7ii.jpg)      
 和TE关系：      
 ![NO2&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_4.png)      

## LSTV：
 ![](http://ww1.sinaimg.cn/large/006y8lVajw1fbkboz8omvj31ke0yinbl.jpg)      
 和TE关系：	      
 ![Light&TE](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_5.png)      

## TE
 ![TE](http://ww4.sinaimg.cn/large/006y8lVajw1fbkbgj08xxj31hg0yy7f0.jpg)    
 去除TE=0，41165行，无空值。    
 前面两个图，分别是box和violin，可以通过这个图看到数据的分布情况。和后面第7点的分析一样，这一列的数据尾部差异特别大。    
 
## 数据分析结论
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_12.png)
 1. TE列，可以看到最后一个四分位数的差异和前面的差了一个数量级。
 2. NPP列，总体来看，差异可以接受。
 3. Light，最后一个四分位数的差异也很大。
 4. SO2,总体来看，差异可以接受,但明显第一个四分位数有大量0值。
 5. NO2、LSTV的差异也是一样，可以接受。
 6. 对于差异相对较大的数据，这样的跨度可能是两种情况，一是部分数据是异常点，二是这些数据的分布在尾部变得稀疏，需要进一步考虑。如果是异常点，将在此数据的基础上训练模型，判断模型预测错误的情况，是否与这些异常有关。如果确实是这样，可以采取步骤进行矫正。比如，可以复制这些预测模型表现不好的例子，以提高其在训练集中的比例；或者，把不好的例子分离出来，然后单独训练；当然也可以把这些例子去除。
 7. 整体数据可视化结果：      
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_13.png)       
 可以看到，深蓝线（TE值较高）聚集在Light的高值区域；而黄线聚集在低值区域。但数据特点并不是特别明显，很多数据没有明显的相关性,黄线聚集在属性的全部区域。

# 线性回归

## 数据相关性：
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_6.png)    
 红色表示相关，蓝色表示无关，下面是实际数据值。
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/cor.png)
## 使用有相对显著相关性的Light计算结果：
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_7.png)
## +SO2,NO2
 泛化结果0.092，这是结果比较好的。
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_8.png)
## 所有X
 后面的数据都使用了所有的属性来计算，感觉特征太少了，加入更多特征更有利。
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_9.png)

# 贝叶斯
 贝叶斯的结果虽然不好，但是它的泛化能力不错。   
 泛化结果0.092.     
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_10.png)

# 多项式+线性
 把每个属性取了平方、三次方、四次方、5次方，然后用线性回归做的结果。    
 效果还是有提升的。      
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_11.png)

# 随机森林
 决策树数目和其对应的RMSE变化曲线：      
 <!--![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_14.png)      -->
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_20.png)      
 每个参数的重要性：      
 <!--![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_15.png)      -->
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_21.png)      
 使用最佳参数得到的结果，决策树的结果还是比较好的，但是泛化能力**很差**,0.0424：      
 ![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_16.png)    
 
 多项式+随机森林，泛化结果：0.076

# 梯度提升
 和随机森林一样，泛化能力差。     
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_19.png) 

# 其他      
 取log，效果很差      

# 聚类
## 只用xy聚类
聚类的总数参数选择：   
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_22.png)   

聚类结果：
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_23.png)   

+线性：0.113
+贝叶斯：0.021
+随机森林：0.1219
+梯度提升：-0.086
+多项式&线性：-96
+多项式&随机森林：0.136
+多项式&贝叶斯：太差
+贝叶斯&boosting：差
+决策树&boosting：差

##用除了TE以外的数聚类：
参数选择：   
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_24.png)

聚类结果：
+线性：0.126
+贝叶斯：0.027
+随机森林：0.116
+梯度提升：差
+多项式&线性：差
+多项式&随机森林：0.08
+多项式&贝叶斯：差
+贝叶斯&boosting：差
+决策树&boosting：差

##用TE聚类：
参数选择：
![](https://raw.githubusercontent.com/Moirai7/environment/master/pic/figure_25.png)

聚类结果：
+线性：0.95
+贝叶斯：0.98
+随机森林：0.98
+梯度提升：0.98
+多项式&线性：差
+多项式&随机森林：0.982
+多项式&贝叶斯：
+贝叶斯&boosting：
+决策树&boosting：0.97

----
 TODO：   
 1. 尝试多次方+随机森林(done)
 2. 试一下bagging(done)
 3. 用地理位置X,Y聚类，每个聚类分别做回归试一下
 4. 用所有的数据聚类，再试一下回归
 
 
 
 
 
 
 
 
 
 
 
