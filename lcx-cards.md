# 卡片
[TOC]
#####206. Validation Curve 验证曲线
-by lcx

![206](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-206.jpg)
1. 验证曲线的作用？
我们知道误差由偏差(bias)、方差(variance)和噪声(noise)组成。

- 偏差：模型对于不同的训练样本集，预测结果的平均误差。

- 方差：模型对于不同训练样本集的敏感程度。

- 噪声：数据集本身的一项属性。

同样的数据（cos函数上的点加上噪声），我们用同样的模型（polynomial），但是超参数却不同（degree ＝ 1, 4 ，15），会得到不同的拟合效果：

![验证曲线——1](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/验证曲线_1.png)

- 第一个模型太简单，模型本身就拟合不了这些数据（高偏差）；

- 第二个模型可以看成几乎完美地拟合了数据；

- 第三个模型完美拟合了所有训练数据，但却不能很好地拟合真实的函数，也就是对于不同的训练数据很敏感（高方差）。

对于这两个问题，我们可以选择模型和超参数来得到效果更好的配置，也就是可以通过验证曲线调节。

2. 验证曲线与学习曲线
**验证曲线和学习曲线的区别是：**

- 横轴为某个超参数的一系列值，由此来看不同参数设置下模型的准确率，而不是不同训练集大小下的准确率。

- 从验证曲线上可以看到随着超参数设置的改变，模型可能从欠拟合到合适再到过拟合的过程，进而选择一个合适的设置，来提高模型的性能。

- 需要注意的是如果我们使用验证分数来优化超参数，那么该验证分数是有偏差的，它无法再代表模型的泛化能力，我们就需要使用其他测试集来重新评估模型的泛化能力。

- 不过有时画出单个超参数与训练分数和验证分数的关系图，有助于观察该模型在相应的超参数取值时，是否有过拟合或欠拟合的情况发生。

（1） 使用学习曲线判别偏差和方差问题

如果一个模型相对于训练集来说过于复杂，比如参数太多，则模型很可能过拟合。避免过拟合的手段包含增大训练集，但这是不容易做到的。通过画出不同训练集大小对应的训练集和验证集准确率，我们能够很轻松滴检测模型是否方差偏高或偏差过高，以及增大训练集是否有用。

![验证曲线——2](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/验证曲线_2.png)
- 上图的左上角子图中模型偏差很高。它的训练集和验证集准确率都很低，很可能是欠拟合。解决欠拟合的方法就是增加模型参数，比如，构建更多的特征，减小正则项。

- 上图右上角子图中模型方差很高，表现就是训练集和验证集准确率相差太多。解决过拟合的方法有增大训练集或者降低模型复杂度，比如增大正则项，或者通过特征选择减少特征数。

这俩问题可以通过验证曲线解决。
我们先看看学习曲线是怎么回事吧：

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import numpy as np
from sklearn.learning_curve import validation_curve
#导入数据
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)#类标整数化
print (le.transform(['M','B']))
#划分训练集合测试集
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.20,random_state=1)
#标准化、模型训练串联
pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state=1,penalty='l2'))])


case1：学习曲线
构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)
统计结果
train_mean= np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
绘制效果
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='test accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()
```
![验证曲线——3](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/验证曲线_3.jpg)
(2) 用验证曲线解决过拟合和欠拟合

验证曲线是非常有用的工具，他可以用来提高模型的性能，原因是他能处理过拟合和欠拟合问题。

验证曲线和学习曲线很相近，不同的是这里画出的是不同参数下模型的准确率而不是不同训练集大小下的准确率：


```
case2：验证曲线
param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
10折，验证正则化参数C
train_scores,test_scores =validation_curve(estimator=pipe_lr,X=X_train,y=y_train,param_name='clf__C',param_range=param_range,cv=10)
统计结果
train_mean= np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='test accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()
```
![验证曲线——4](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/验证曲线_4.jpg)

- 我们得到了参数C的验证曲线。

- 和learningcurve方法很像，validationcurve方法使用采样k折交叉验证来评估模型的性能。在validation_curve内部，我们设定了用来评估的参数，这里是C,也就是LR的正则系数的倒数。

- 观察上图，最好的C值是0.1。

####?损失函数
1. 概述
- 通常机器学习每一个算法中都会有一个目标函数，算法的求解过程是通过对这个目标函数优化的过程。在分类或者回归问题中，通常使用损失函数（代价函数）作为其目标函数。损失函数用来评价模型的预测值和真实值不一样的程度，损失函数越好，通常模型的性能越好。不同的算法使用的损失函数不一样。

- 损失函数分为经验风险损失函数和结构风险损失函数。经验风险损失函数指预测结果和实际结果的差别，结构风险损失函数是指经验风险损失函数加上正则项。通常表示为如下：



2. 常见的损失函数
- 0-1损失函数和绝对值损失函数
0-1损失是指，预测值和目标值不相等为1，否则为0：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_1.png)

感知机就是用的这种损失函数。但是由于相等这个条件太过严格，因此我们可以放宽条件，即满足![损失函数_2]() 时认为相等。

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_3.png)
绝对值损失函数为：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_4.png)

- log对数损失函数

逻辑斯特回归的损失函数就是对数损失函数，在逻辑斯特回归的推导中，它假设样本服从伯努利分布（0-1）分布，然后求得满足该分布的似然函数，接着用对数求极值。逻辑斯特回归并没有求对数似然函数的最大值，而是把极大化当做一个思想，进而推导它的风险函数为最小化的负的似然函数。从损失函数的角度上，它就成为了log损失函数。

log损失函数的标准形式：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_5.png)

在极大似然估计中，通常都是先取对数再求导，再找极值点，这样做是方便计算极大似然估计。损失函数L（Y，P（Y|X））是指样本X在分类Y的情况下，使概率P(Y|X)达到最大值（利用已知的样本分布，找到最大概率导致这种分布的参数值）

- 平方损失函数

最小二乘法是线性回归的一种方法，它将回归的问题转化为了凸优化的问题。最小二乘法的基本原则是：最优拟合曲线应该使得所有点到回归直线的距离和最小。通常用欧几里得距离进行距离的度量。平方损失的损失函数为：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_6.png)

- 指数损失函数

AdaBoost就是一指数损失函数为损失函数的。   指数损失函数的标准形式：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_7.png)

- Hinge损失函数  Hinge loss用于最大间隔（maximum-margin）分类，其中最有代表性的就是支持向量机SVM。Hinge函数的标准形式：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_8.png)　　

（与上面统一的形式：![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_9.png) ）
其中，t为目标值（-1或+1），y是分类器输出的预测值，并不直接是类标签。其含义为，当t和y的符号相同时（表示y预测正确）并且|y|≥1时，hinge loss为0；当t和y的符号相反时，hinge loss随着y的增大线性增大。
在支持向量机中，最初的SVM优化的函数如下：

 ![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_10.png)

将约束项进行变形，则为：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_11.png)

则损失函数可以进一步写为：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_12.png)

因此，SVM的损失函数可以看做是L2正则化与Hinge loss之和。

几种损失函数的曲线如下图：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/损失函数_13.png)
参考文献
[常见的损失函数](https://www.cnblogs.com/hejunlin1992/p/8158933.html)

####?范数

什么是范数？

我们知道距离的定义是一个宽泛的概念，只要满足非负、自反、三角不等式就可以称之为距离。范数是一种强化了的距离概念，它在定义上比距离多了一条数乘的运算法则。有时候为了便于理解，我们可以把范数当作距离来理解。

在数学上，范数包括向量范数和矩阵范数，向量范数表征向量空间中向量的大小，矩阵范数表征矩阵引起变化的大小。一种非严密的解释就是，对应向量范数，向量空间中的向量都是有大小的，这个大小如何度量，就是用范数来度量的，不同的范数都可以来度量这个大小，就好比米和尺都可以来度量远近一样；对于矩阵范数，学过线性代数，我们知道，通过运算AX=B
，可以将向量X变化为B，矩阵范数就是来度量这个变化大小的。

向量范数
1. L-P范数
与闵可夫斯基距离的定义一样，L-P范数不是一个范数，而是一组范数，其定义如下：
    $Lp=\sqrt[p]{\sum\limits_{1}^n  x_i^p}，x=(x_1,x_2,\cdots,x_n)$

    根据P 的变化，范数也有着不同的变化，一个经典的有关P范数的变化图如下：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/范数_1.png)

    上图表示了p从无穷到0变化时，三维空间中到原点的距离（范数）为1的点构成的图形的变化情况。以常见的L-2范数（p=2）为例，此时的范数也即欧氏距离，空间中到原点的欧氏距离为1的点构成了一个球面。
    实际上，在0≤p<1时，Lp并不满足三角不等式的性质，也就不是严格意义下的范数。以p=0.5，二维坐标(1,4)、(4,1)、(1,9)为例，$\sqrt[0.5]{(1+\sqrt{4})}+\sqrt[0.5]{(\sqrt{4}+1)}<\sqrt[0.5]{(1+\sqrt{9})}$。因此这里的L-P范数只是一个概念上的宽泛说法。
2. L0范数
当P=0时，也就是L0范数，由上面可知，L0范数并不是一个真正的范数，它主要被用来度量向量中非零元素的个数。用上面的L-P定义可以得到的L-0的定义为：
    $||x||=\sqrt[0]{\sum\limits_1^nx_i^0}，x=(x_1,x_2,\cdots,x_n)$
    这里就有点问题了，我们知道非零元素的零次方为1，但零的零次方，非零数开零次方都是什么鬼，很不好说明L0的意义，所以在通常情况下，大家都用的是：
    $||x||_0=$#$(i|x_i\neq 0)$
    表示向量x中非零元素的个数。
    对于L0范数，其优化问题为：
    $<min||x||_0$
    在实际应用中，由于L0范数本身不容易有一个好的数学表示形式，给出上面问题的形式化表示是一个很难的问题，故被人认为是一个NP难问题。所以在实际情况中，L0的最优问题会被放宽到L1或L2下的最优化。
3. L1范数
L1范数是我们经常见到的一种范数，它的定义如下：
    $||x||_1=\sum_i|x_i|$
    表示向量x中非零元素的绝对值之和。
    L1范数有很多的名字，例如我们熟悉的曼哈顿距离、最小绝对误差等。使用L1范数可以度量两个向量间的差异，如绝对误差和（Sum of Absolute Difference）：
    $SAD(x_1,x_2)=\sum_i|x_{1i}-x_{2i}|$
    对于L1范数，它的优化问题如下：
    $min ||x||_1$
     s.t.Ax=b
    由于L1范数的天然性质，对L1优化的解是一个稀疏解，因此L1范数也被叫做稀疏规则算子。通过L1可以实现特征的稀疏，去掉一些没有信息的特征，例如在对用户的电影爱好做分类的时候，用户有100个特征，可能只有十几个特征是对分类有用的，大部分特征如身高体重等可能都是无用的，利用L1范数就可以过滤掉。
4. L2范数
L2范数是我们最常见最常用的范数了，我们用的最多的度量距离欧氏距离就是一种L2范数，它的定义如下：
    $||x||_2=\sqrt{\sum_ix_i^2}$
    表示向量元素的平方和再开平方。
    像L1范数一样，L2也可以度量两个向量间的差异，如平方差和（Sum of Squared Difference）:
    $SSD(x_1,x_2)=\sum_i(x_{1i}-x_{2i})^2$
    对于L2范数，它的优化问题如下：
    $min ||x||_2$
    s.t.Ax=b
    L2范数通常会被用来做优化目标函数的正则化项，防止模型为了迎合训练集而过于复杂造成过拟合的情况，从而提高模型的泛化能力。
5. L-∞范数
当P=∞时，也就是L-∞范数，它主要被用来度量向量元素的最大值。用上面的L-P定义可以得到的L∞的定义为：
$||x||_\infty=\sqrt[\infty]{\sum\limits_1^nx_i^\infty}，x=(x_1,x_2,\cdots,x_n)$
与L0一样，在通常情况下，大家都用的是：
$<script id="MathJax-Element-12449" type="math/tex">||x||_\infty=max(|x_i|)</script>$
来表示L∞

参考文献
[几种范数的简单介绍](https://blog.csdn.net/shijing_0214/article/details/51757564)

####?贪心算法（greedy）
1. 定义
只为下一个即时决定进行优化

贪心算法是指在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，只做出在某种意义上的局部最优解。贪心算法不是对所有问题都能得到整体最优解，关键是贪心策略的选择，选择的贪心策略必须具备无后效性，即某个状态以前的过程不会影响以后的状态，只与当前状态有关。

解题的一般步骤是：

- 建立数学模型来描述问题；
- 把求解的问题分成若干个子问题；
- 对每一子问题求解，得到子问题的局部最优解；
把子问题的局部最优解合成原来问题的一个解。

2. 基本思想
贪心算法是最直观的算法设计范式之一。利用贪心算法的求解方式与递归调用极为相似，都是先把问题分割成几个子问题，并在每个阶段生成一部分答案。

从这一点（原问题 ⇒ 多个规模更小的子问题）上看贪心算法和穷举搜索算法以及动态规划算法并无太大区别。不过，与“先考虑所有选项，然后再找出最优解（也即并不确定当前阶段哪个结果对应于最好的结果）”的穷举搜索算法和动态规划算法不同的是，贪心算法在每个阶段即可找出当前最优解，贪心算法也不会考虑当前选择对以后选择的影响。
穷举/动态 ⇒ 全局
贪心 ⇒ 局部
这也就造成了，很多情况下，贪心都无法求出最优解。因此，贪心法的使用范围主要限制在以下两种情况：

- 即使使用贪心法也能求出最优解。贪心法比动态规划算法具有更快的运算速度，故在这种情况下，贪心算法会十分有优势。

- 因时间和空间限制而无法利用其它算法求出最优解时，可利用近似解替代最优解。这种情况下，利用贪心法求出的解虽然不是最优解，但比其他答案更接近最优解。

3. 举例
例子：
[背包问题]

    有一个背包，背包容量是M=150。有7个物品，物品可以分割成任意大小。 要求尽可能让装入背包中的物品总价值最大，但不能超过总容量。
      物品 A B C D E F G
      重量 35 30 60 50 40 10 25
      价值 10 40 30 50 35 40 30
- 分析：
  - 目标函数： ∑pi最大
  - 约束条件：装入的物品总重量不超过背包容量，即∑wi<=M( M=150)

（1） 根据贪心的策略，每次挑选价值最大的物品装入背包，得到的结果是否最优？
（2） 每次挑选所占重量最小的物品装入是否能得到最优解？
（3） 每次选取单位重量价值最大的物品，成为解本题的策略?

对于本例题中的3种贪心策略，都无法成立，即无法被证明，解释如下：

（1）贪心策略：选取价值最大者。
反例：
W=30
物品：A B C
重量：28 12 12
价值：30 20 20
根据策略，首先选取物品A，接下来就无法再选取了，可是，选取B、C则更好。
（2）贪心策略：选取重量最小。它的反例与第一种策略的反例差不多。
（3）贪心策略：选取单位重量价值最大的物品。反例：
W=30
物品：A B C
重量：28 20 10
价值：28 20 10
根据策略，三种物品单位重量价值一样，程序无法依据现有策略作出判断，如果选择A，则答案错误。
值得注意的是，贪心算法并不是完全不可以使用，贪心策略一旦经过证明成立后，它就是一种高效的算法。比如，求最小生成树的Prim算法和Kruskal算法都是漂亮的贪心算法。



在深度学习中的应用
梯度下降算法
决策树，它的每一次节点分裂也都是贪心的

参考文献
[从零开始学贪心算法](https://blog.csdn.net/qq_32400847/article/details/51336300)

####?分类
分类是机器学习、模式识别中很重要的一环，就是，因为计算机其实无法深层次地理解文字图片目标的
意思，只能回答是或者不是

分类和回归的区别:
主要在于输出变量的类型。定量输出称为回归，或者说是连续变量预测;定性输出称为分类，或者说是
离散变量预测。举个例子:预测明天的气温是多少度，这是一个回归任务;预测明天是阴、晴还是雨，
就是一个分类任务。

- 二者常见算法的比较:
    1. Logistic Regression 和 Linear Regression:
Linear Regression: 输出一个标量 wx+b，这个值是连续值，所以可以用来处理回归问题。
Logistic Regression:把上面的 wx+b 通过 sigmoid函数映射到(0,1)上，并划分一个阈值，大于阈值的分为 一类，小于等于分为另一类，可以用来处理二分类问题。 更进一步:对于N分类问题，则是先得到N组w 值不同的 wx+b，然后归一化，比如用 softmax函数，最后变成N个类上的概率，可以处理多分类问题。
    2. Support Vector Regression 和 Support Vector Machine:
SVR:输出 wx+b，即某个样本点到分类面的距离，是连续值，所以是回归模型。 SVM:把这个距离用 sign(·) 函数作用，距离为正(在超平面一侧)的样本点是一类，为负的是另一类，所 以是分类模型。
    3. 神经网络用于分类和回归:
用于回归:最后一层有m个神经元，每个神经元输出一个标量，m个神经元的输出可以看做向量 v，现全 部连到一个神经元上，则这个神经元输出wv+b，是一个连续值，可以处理回归问题，跟上面 Linear Regression 思想一样。
用于N分类:现在这m个神经元最后连接到 N 个神经元，就有 N 组w值不同的 wv+b，同理可以归一化 (比如用 softmax )变成N个类上的概率。

常用的分类算法:
  - 朴素贝叶斯
  - 决策树
  - KNN算法

#####225. ACCURACY 准确率

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-225.jpg)
这是一个常见的分类标准，常用的评价分类好坏的四个标准 首先假设原始样本中有两类，其中:

1. 总共有 P个类别为1的样本，假设类别1为正例。
2. 总共有N个类别为0的样本，假设类别0为负例。
3. 有 TP个类别为1的样本被系统正确判定为类别1，FN 个类别为1 的样本被系统误判定为类别 0，显然 有P=TP+FN。
4. 有 FP 个类别为0 的样本被系统误判断定为类别1，TN 个类别为0 的样本被系统正确判为类别 0，显然有N=FP+TN;
其他
- 精确度(Precision):

    P = TP/(TP+FP) ; 反映了被分类器判定的正例中真正的正例样本的比重。
- 准确率(Accuracy):

    A = (TP + TN)/(P+N) = (TP + TN)/(TP + FN + FP + TN); 反映了分类器统对整个样本的判定能力——能将 正的判定为正，负的判定为负 。
- 召回率(Recall)也称为 True Positive Rate:

    R = TP/(TP+FN) = 1 - FN/T; 反映了被正确判定的正例占总的正例的比重 。 F1 measure (F-measure or balanced F-score):
F = 2 * 召回率 * 准确率/ (召回率+准确率);。
参考文献
[关于准确率，精确率，召回率，f-值等](https://blog.csdn.net/qq_24343273/article/details/51992064)



?向前逐步选择
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15217731785506.png)

翻译
1. 创建一个没有特征和预测的模型m0
2. 循环f次

* 创建一个模型，将一个特性添加到现有模型中。
* 重复所有未使用的特性。
* 选择具有最佳评价指标的模型（像RSS或者R平方），并定义为mk+1。

3. 在所有模型里，使用交叉验证和交叉评价测度模型来选择最佳的


定义
*     向前选择法是从判别模型中没有变量开始，每一步把一个队判别模型的判断能力贡献最大的变量引入模型，直到没有被引入模型的变量都不符合进入模型的条件时，变量引入过程结束。当希望较多变量留在判别函数中时，使用向前选择法
2.     向后选择法与向前选择法完全相反。它是把用户所有指定的变量建立一个全模型。每一步把一个对模型的判断能力贡献最小的变量剔除模型，知道模型中的所用变量都不符合留在模型中的条件时，剔除工作结束。在希望较少的变量留在判别函数中时，使用向后选择法。
3.     逐步选择法是一种选择最能反映类间差异的变量子集，建立判别函数的方法。它是从模型中没有任何变量开始，每一步都对模型进行检验，将模型外对模型的判别贡献最大的变量加入到模型中，同时也检查在模型中是否存在“由于新变量的引入而对判别贡献变得不太显著”的 变量，如果有，则将其从模型中出，以此类推，直到模型中的所有变量都符合引入模型的条件，而模型外所有变量都不符合引入模型的条件为之，则整个过程结束


参考文献
[逐步回归](https://wenku.baidu.com/view/a288035d67ec102de3bd8939.html)


####?稀疏性
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15217732169127.png)
翻译：
稀疏矩阵可以存储更多有效的数据，尤其是有很多0和缺失值的情况下

定义

* 稀疏矩阵是一个几乎由零值组成的矩阵。稀疏矩阵与大多数非零值的矩阵不同，非零值的矩阵被称为稠密矩阵。
* 如果矩阵中的许多系数都为零，那么该矩阵就是稀疏的。对稀疏现象有兴趣是因为它的开发可以带来巨大的计算节省，并且在许多大的实践中都会出现矩阵稀疏的问题。
* 矩阵的稀疏性可以用一个得分来量化，也就是矩阵中零值的个数除以矩阵中元素的总个数。
* sparsity= count zeroelements/ totalelements
机器学习中的稀疏矩阵

* 机器学习中的一些领域必须开发专门的方法来解决稀疏问题，因为输入的数据几乎总是稀疏的。
* 三个例子包括:

    1. 用于处理文本文档的自然语言处理。
    2. 推荐系统在一个目录中进行产品使用。
    3. 当处理图像时计算机视觉包含许多黑色像素（black pixel）。

* 如果在语言模型中有100,000个单词，那么特征向量长度为100,000，但是对于一个简短的电子邮件来说，几乎所有的特征都是0。*

稀疏矩阵的python实现
* python中scipy模块中，有一个模块叫sparse模块，就是专门为了解决稀疏矩阵而生。
 [稀疏矩阵的python实现](http://blog.csdn.net/bitcarmanlee/article/details/52668477)



####?预处理训练集和测试集
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15217748933397.png)

翻译：
1. 先将预处理器处理训练集数据。
2. 将其应用于训练集和测试集。
为什么呢？因为我们必须假装测试集是未知的数据。

预处理
数据预处理一般包括：

1. 数据标准化

    这是最常用的数据预处理，把某个特征的所有样本转换成均值为0，方差为1。
    其中，可以调用sklearn.preprocessing中的StandardScaler()进行数据的标准化。
2. 数据归一化

    把某个特征的所有样本取值限定在规定范围内（一般为[-1,1]或者[0,1]）。
归一化得方法为：
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15217747860685.png)

    可以调用sklearn.preprocessing中的MinMaxScaler()将数据限定在[0,1]范围，调用MaxAbsScaler()将数据限定在[-1,1]范围。
3. 数据正规化

    把某个特征的所有样本的模长转换为1。方法为：
    ![-w140](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15217748071873.png)

    可以调用sklearn.preprocessing中的Normalizer()实现
4. 数据二值化
    把数据的特征取值根据阈值转为为0或者1。
5. 数据缺值处理
    对于缺失的特征数据，进行数据填补，一般填补的方法有：均值，中位数，众数填补等。
6. 数据离群点处理
    删除离群点数据。
7. 数据类型转换
    如果数据的特征不是数值型特征，则需要转换为数值型。

举例
* **问题：数据预处理的归一化手段应该如何应用到训练集，测试集和验证集中？**

    *假如先把数据划分成训练集和测试集，我在训练集上对所有变量归一化后，比如用均值方差归一化，那我在测试集上归一化的时候用的均值方差都是训练集中的还是在测试集上用自身的均值方差。*

* **解答：**测试集的归一化的均值和标准偏差应该来源于训练集。如果熟悉Python的sklearn的话，就应该知道应先对训练集数据fit，得到包含均值和标准偏差的scaler，然后再分别对训练集和验证集transform。最容易犯的错误就是先归一化，再划分训练测试集。


    当我们对训练集应用各种预处理操作时（特征标准化、主成分分析等等），我们都需要对测试集重复利用这些参数。



#####130. R^2
-by lcx

![130](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-130.jpg)

1. 定义
1. Sum Of Squares Due To Error 误差平方和

    ![](http://img.blog.csdn.net/20160621234036994?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
    对于第i个观察点, 真实数据的Yi与估算出来的Yi-head的之间的差称为第i个residual, SSE 就是所有观察点的residual的和

2. Total Sum Of Squares 总平方和

    ![](http://img.blog.csdn.net/20160621234113695?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

3. Sum Of Squares Due To Regression 回归平方和

    ![](http://img.blog.csdn.net/20160621234146760?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
4. 通过以上我们能得到以下关于他们三者的关系

    ![](http://img.blog.csdn.net/20160621234237448?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

2. 决定系数与相关系数

- 决定系数：R^2 判定系数就是拟合优度判定系数，它体现了回归模型中自变量的变异在因变量的变异中所占的比例。如R^2 =0.99999表示在因变量y的变异中有99.999%是由于变量x引起。当 R^2 =1时表示，所有观测点都落在拟合的直线或曲线上；当R^2 =0时，表示自变量与因变量不存在直线或曲线关系。


- 相关系数：测试因变量和自变量他们之间的线性关系有多强，也就是说, 自变量产生变化时因变量的变化有多大，可以反映是正相关还是负相关



#####208. VIF 方差膨胀因子
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-208.jpg)

1. 翻译
- 它测量了特征之间的共线效应。
- 它衡量的是模型参数(系数)的方差增加的程度。
- 为了计算它，我们使一个特性成为模型的目标(而不是依赖变量)。运行该模型，然后计算R^2。

2. 定义

- 在统计学中，**方差膨胀因子(VIF)在一个普通最小二乘回归分析中量化了多重共线性的严重程度**。它提供了一个指数，用来衡量估计回归系数的方差(估计的标准差的平方)由于共线性而增加的程度。

- 方差膨胀因子允许快速测量变量对回归中标准错误的贡献程度。当存在显著的多重共线性问题时，所涉及的变量的方差膨胀系数将非常大。在确定了这些变量之后，有几种方法可以用来消除或合并共线变量，从而解决多重共线性问题。

- 分析多重共线性的大小通过考虑VIF的大小,一个共同的经验法则是：
    VIF<=1时，无多重共线性；
    1<VIF<=5时，多重共线性适中；
    5<VIF<=10时，多重共线性高。
3. 多重共线性

- 我们进行回归分析需要了解每个自变量对因变量的单纯效应，多重共线性就是说自变量间存在某种函数关系。
- 如果你的两个自变量间（X1和X2）存在函数关系，那么X1改变一个单位时，X2也会相应地改变，此时你无法做到固定其他条件，单独考查X1对因变量Y的作用，你所观察到的X1的效应总是混杂了X2的作用，这就造成了分析误差，使得对自变量效应的分析不准确，所以做回归分析时需要排除多重共线性的影响

#####180. T统计量
-by lcx
![180](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-180.jpg)
1. 翻译
t统计量度量标准偏差Bi的数量远离常数。如果Bi=0，我们可以计算得到T分布等于或低的概率。这叫P-score
2. 定义
- T-statistic是根据model计算的,用来做检验的统计量.
正常T-statistic应该在0假设(null hypothesis)为真时,服从T分布(T-distribution).
    T-test时根据T-statistic值的大小计算p-value,决定是接受还是拒绝假设.

- t检验
    1. 点估计值为b（点估计是利用样本数据对未知参数进行估计得到的是一个具体的数据），b的抽样分布服从t分布，且抽样分布标准差为SE
    2. 现在我们假设b对应的总体参数值为a，若b是a的无偏点估计有E(b)=a
    3. 计算t值：t=（b-a）/ SE

- t值所表达的意思是，在样本统计量b抽样分布为t分布情况下，点估计值b和我们假设的总体参数a之间差异，这个差异以抽样分布标准差为尺度。

- 综上，t值越小说明在样本统计量抽样分布服从t分布情况下，点估计值b和研究者假设的总体参数a之间的差异相对于抽样分布标准差而言越小，也就是点估计b和假设的总体参数a的区别不显著，是一致的，可以接受假设a；t值越大则反之。当然实际使用时还需要根据t值去查t分布表，根据显著度来判断是否接受假设a。


#####82. 自然对数
![1041521813190_.pic_hd](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/1041521813190_.pic_hd.jpg)
1. why
* 为什么需要做数据变换？

    * 从直观上讲，是为了更便捷的发现数据之间的关系（可以理解为更好的数据可视化）。

    * 举个栗子，下图的左图是各国人均GDP和城市人口数量的关系，可以发现人均GDP是严重左偏的，并且可以预知在回归方程中存在明显的异方差性，但如果对GDP进行对数变换后，可以发现较明显的线性关系。
    * 为什么呢？因为我们度量相关性时使用的Pearson相关系数检验的是变量间的线性关系，只有两变量服从不相关的二元正态分布时，Pearson相关系数才会服从标准的t-分布，但如果变量间的关系是非线性的，则两个不独立的变量之间的Pearson相关系数也可以为0.

    ![](https://pic2.zhimg.com/80/3933be3a92533b7fe3b210e67da2c077_hd.jpg)

    * 所以，数据变换后可以更便捷的进行统计推断(t检验、ANOVA或者线性回归分析)。
    * 另外，最经典的例子就是回归分析中的异方差性，误差项的方差随着自变量的变化而变化，如果直接进行回归估计残差的方差会随着自变量的变化而变化，如果对变量进行适当变换，此时残差服从同一个正态分布。

* 为什么可以做数据变换？

    * 每当做数据变换时，禁不住会想这样原始的数据信息是否经过变换后存在损失？数据变换有没有标准程序？原始数据的统计推断又该怎么进行？

    * 先从理论情形下去考虑:

    1.  例子1，如果一个数是连续的，并且服从对数正态分布![](https://www.zhihu.com/equation?tex=ln%28X%29%5Csim+N%28%5Cmu%2C%5Csigma%5E%7B2%7D%29)，可以很容易知道的概率密度函数(PDF)![](https://www.zhihu.com/equation?tex=f_%7BX%7D%28x%29%3D%5Cfrac%7B1%7D%7Bx%5Csqrt%7B2%5Cpi%7D%5Csigma%7D+e%5E%7B-%5Cfrac%7B%28lnx-%5Cmu%29%5E2%7D%7B2%5Csigma%5E%7B2%7D%7D+%7D+)，这样![](https://www.zhihu.com/equation?tex=E%28X%29%3De%5E%7B%5Cmu%2B%5Cfrac%7B%5Csigma%5E%7B2%7D%7D%7B2%7D%7D)![](https://www.zhihu.com/equation?tex=Var%28X%29%3D%5Cleft%28+e%5E%7B%5Csigma%7B2%7D-1%7D++%5Cright%29e%5E%7B2%5Cmu%2B%5Csigma%5E2%7D+)，此时可以看到已知变换后的数据的统计特征可以反过来推导出原始数据的统计特征，不存在数据信息的损失（可以看到对数转换后变量的均值可以直接由样本数据的均值得到，但不进行变化却需要由样本均值方差两方面去推断得到）；

    1.  例子2，如果一个数是离散的，服从负二项分布，概率质量函数(PMF)可以写成![](https://www.zhihu.com/equation?tex=f%5Cleft%28+k%3A%5Ctheta%2Cp%5Cright%29%3DC_%7B%5Ctheta-1%7D%5E%7Bk%2B%5Ctheta-1%7D+%5Cleft%28+1-p+%5Cright%29%5E%5Ctheta+p%5Ek)，如果对这个变量进行对数变换后，情形又会怎样呢？此时![](https://www.zhihu.com/equation?tex=E%5Cleft%28+k+%5Cright%29+%3D%5Cfrac%7Bp%5E%7B%5Ctheta%7D%7D%7B1-p%7D+)![](https://www.zhihu.com/equation?tex=Var%5Cleft%28+k+%5Cright%29%3D%5Cfrac%7Bp%5E%5Ctheta%7D%7B%281-p%29%5E2%7D+)，假设数据的生成过程服从负二项分布，并且在不同的![](https://www.zhihu.com/equation?tex=%5Ctheta%3D0.5%2C1%2C2%2C5%2C10%2C100)下模拟生成数据，再用不同的方式去估计![](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmu%7D%3DY+)，可以设计评价指标![](https://www.zhihu.com/equation?tex=B%3D%5Cfrac%7B1%7D%7BS%7D%5CSigma_%7B%5Cleft%28s%5Cright%29%7D+%5Ctilde%7B%5Cmu%7D+-%5Cmu)，可以看到不同的数据变换方式下的估计精度是不同的。可以看到，如果假设数据服从负二项分布，估计的误差很小，如果假设数据对数变换后服从正态分布时会出现较大误差（由于离散分布时数据可以取0，此时对数变换需要用![](https://www.zhihu.com/equation?tex=ln%5Cleft%28+x%2Bk+%5Cright%29+)的形式，可以发现的取值并非随意），如果假设数据根号变换后服从正态分布时的误差要小于对数变换。所以，从一个小的随机模拟实验可以看出，数据的变换方式并非随意，并且对数变换也不一定是最好的变换，尤其是离散数据情况下

    ![](https://pic2.zhimg.com/80/e722a04629f0750c4d460a0e52523952_hd.jpg)

    * 但上述仅仅是在理论前提下数据变换的讨论，但实际应用中呢？理论前提下，即使再复杂总能找到处理的办法，但应用问题却没有标准答案。在我看来，数据变换方法的使用更是一门艺术（先验知识+经验+运气），需要结合应用领域的专门知识。

* 总结：采用对数描述变量，一是变化率的问题。二是用对数能够描述较大的动态范围。三是符合人的心理感知特性。

参考文献
[在统计学中为什么要对变量取对数？-知乎] (https://www.zhihu.com/question/22012482/answer/21357107)

#####118. 感知机
-by lcx
![1051521813191_.pic_hd](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/1051521813191_.pic_hd.jpg)
1. 形象的感知机
* 什么是感知机模型？

    简而言之就是可以将有两个特征的数据集中的正例和反例完全分开的关于一条直线的函数，或者可以是关于三维空间中的一个平面的函数，更可以是关于高维的一个超平面的函数。

    其实感知机就是一个分段函数，不过变量是一条直线、平面或超平面而已。

    ![](https://img-blog.csdn.net/20170312094333631?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQXJ0cHJvZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    或者下面的这个三维空间中的平面将有三个特征的数据集分成了两部分：

    ![](https://img-blog.csdn.net/20170312094508815?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQXJ0cHJvZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
2. 定义

    假设输入空间(特征向量)为X⊆Rn，输出空间为Y={-1, +1}。输入x∈X表示实例的特征向量，对应于输入空间的点；输出y∈Y表示示例的类别。由输入空间到输出空间的函数为：
    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221173595890.jpg)


    称为感知机。其中，参数w叫做权值向量weight，b称为偏置bias。**w·x**表示w和x的点积

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221173093779.jpg)

    sign为符号函数，即

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221173312107.jpg)

    在二分类问题中，f(x)的值（+1或-1）用于分类x为正样本（+1）还是负样本（-1）。感知机是一种线性分类模型，属于判别模型。我们需要做的就是找到一个最佳的满足w⋅x+b=0的w和b值，即分离超平面（separating hyperplane）。如下图，一个线性可分的感知机模型
    ![](http://img.blog.csdn.net/20151005162258500)
    中间的直线即w⋅x+b=0这条直线。

    线性分类器的几何表示有：直线、平面、超平面。
3. 学习策略
**核心：极小化损失函数。**

    如果训练集是可分的，感知机的学习目的是求得一个能将训练集正实例点和负实例点完全分开的分离超平面。为了找到这样一个平面（或超平面），即确定感知机模型参数w和b，我们采用的是损失函数，同时并将损失函数极小化。

    对于损失函数的选择，我们采用的是误分类点到超平面的距离（可以自己推算一下，这里采用的是几何间距，就是点到直线的距离）：

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221189508276.jpg)

    其中||w||是L2范数。

    对于误分类点(xi,yi)来说：

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221189786445.jpg)

    误分类点到超平面的距离为：

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221190089251.jpg)

    那么，所有点到超平面的总距离为：

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221190296397.jpg)

    不考虑![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221190696832.jpg),就得到感知机的损失函数了。

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221190503144.jpg)

    其中M为误分类的集合。这个损失函数就是感知机学习的经验风险函数。

    可以看出，损失函数L(w,b)是非负的。如果没有误分类点，则损失函数的值为0，而且误分类点越少，误分类点距离超平面就越近，损失函数值就越小。同时，损失函数L(w,b)是连续可导函数。
4. 学习算法
* 感知机学习转变成求解损失函数L(w,b)的最优化问题。最优化的方法是随机梯度下降法（stochastic gradient descent），下面给出一个简单的梯度下降的可视化图：
![](http://img.blog.csdn.net/20151005203334645)

    上图就是随机梯度下降法一步一步达到最优值的过程，说明一下，梯度下降其实是局部最优。感知机学习算法本身是误分类驱动的，因此我们采用随机梯度下降法。首先，任选一个超平面w0和b0，然后使用梯度下降法不断地极小化目标函数

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221192906446.jpg)

    极小化过程不是一次使M中所有误分类点的梯度下降，而是一次随机的选取一个误分类点使其梯度下降。使用的规则为 ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221193560148.jpg)，其中α
是步长，∇θℓ(θ)是梯度。假设误分类点集合M是固定的，那么损失函数L(w,b)的梯度通过偏导计算：

    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221194315372.jpg)

- 然后，随机选取一个误分类点，根据上面的规则，计算新的w,b，然后进行更新：
    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/15221194629902.jpg)

    其中η是步长，大于0小于1，在统计学习中称之为学习率（learning rate）。这样，通过迭代可以期待损失函数L(w,b)不断减小，直至为0.

- 算法描述：

    算法：感知机学习算法原始形式

    >输入：T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N，学习速率为η）
输出：w, b;感知机模型f(x)=sign(w·x+b)
(1) 初始化w0,b0，权值可以初始化为0或一个很小的随机数
(2) 在训练数据集中选取（x_i, y_i）
(3) 如果yi(w xi+b)≤0
           w = w + ηy_ix_i
           b = b + ηy_i
(4) 转至（2）,直至训练集中没有误分类点

参考文献
[机器学习-感知机perceptron](https://blog.csdn.net/dream_angel_z/article/details/48915561)
[感知机算法原理（PLA原理）及 Python 实现](https://blog.csdn.net/artprog/article/details/61614452)

















####?普通输出层的激活函数
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/80091526036909_.pic.jpg)
>翻译：二元分类：Sigmiod
>多元分类：Softmax
>回归问题：无激活函数

1. 激活函数
激活函数的主要作用：

* 改变之前数据的线性关系，如果网络中全部是线性变换，则多层网络可以通过矩阵变换，直接转换成一层神经网络，所以激活函数的存在，使得神经网络的“多层”有了实际的意义，使网络更加强大，增加网络的能力，使它可以学习复杂的事物，复杂的数据，以及表示输入输出之间非线性的复杂的任意函数映射。

* 执行数据的归一化，将输入数据映射到某个范围内，再往下传递，这样做的好处是可以限制数据的扩张，防止数据过大导致的溢出风险。

2. 二元分类：sigmoid
（1）首先Sigmoid的公式形式：
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image002.png)
函数图像：
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image004.png)
（2）函数的基本性质：

1.    定义域：（-∞，+∞）

2.    值域：（-1，+1）

3.    函数在定义域内为连续和光滑函数

4.    处处可导，导数为：![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image006.png)

Sigmoid函数之所以叫Sigmoid，是因为函数的图像很想一个字母S。这个函数是一个很有意思的函数，从图像上我们可以观察到一些直观的特性：函数的取值在0-1之间，且在0.5处为中心对称，并且越靠近x=0的取值斜率越大。

（3）对于二分类问题的输出层是Sigmoid函数，这是因为Sigmoid函数可以把实数域光滑的映射到[0,1]空间。函数值恰好可以解释为属于正类的概率（概率的取值范围是0~1）。另外，Sigmoid函数单调递增，连续可导，导数形式非常简单，是一个比较合适的函数

3. 多元分类：softmax
softmax的函数为：
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image008.png)
可以看到它有多个值，所有值加起来刚好等于1，每个输出都映射到了0到1区间，可以看成是概率问题。

把之前softmax的特点性质等扩展进来
4. 回归问题：无激活函数
回归问题不需要激活函数，输出层输出的值即为预测值。

####?凹函数与凸函数
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/80101526036909_.pic.jpg)

**凹函数：除了线段的端点，每条线段都在函数之下。**

**凸函数：除了线段的端点，每条线段都在函数之上。**

在很多机器学习算法中，都会遇到最优化问题。因为我们机器学习算法，就是要在模型空间中找到这样一个模型，使得这个模型在一定范围内具有最优的性能表现。因此，机器学习离不开最优化。然而，对于很多问题，我们并不总能够找到这个最优，很多时候我们都是尽力去找到近似最优，这就是解析解和近似解的范畴。

很多最优化问题都是在目标函数是凸函数或者凹函数的基础上进行的。原因很简单，凸函数的局部极小值就是其全局最小值，凹函数的局部极大值就是其全局最大值。因此，只要我们依据一个策略，一步步地逼近这个极值，最终肯定能够到达全局最值附近。
1.判断目标函数凸或者凹的方法

* 导数计算法：

    计算目标函数的一阶导数和二阶导数。然后作出判断。

    凸函数的一阶充要条件：![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image010.png)
    等号右边是对函数在x点的一阶近似。这个条件的意义是，对于函数在定义域的任意取值，函数的值都大于或者等于对函数在这点的一阶近似。用图来说明就是：
    ![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image012.png)
    通过图可以很清楚地理解这个充要条件，但是，具体在应用中，我们不可能对每一个点都去计算函数的一阶导数吧，因此下面这个充要条件更加实用。

    凸函数的二阶充要条件![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image014.png)
    很简单，如果一个函数的二阶导数大于等于零，那么这个函数就是凸函数。很好理解，函数的一阶导数具有递增性，那么函数本身就是凸函数。

    通过导数计算法，可以很快地判断函数是不是凸函数。凹函数同理。

* 结构分析法：

    有时候我们可以通过分析目标函数的结构，就能在一些情况下判断函数是否是凸函数。下面给出一些结论：

    指数函数是凸函数；

    1.    指数函数是凸函数；

    2.    对数函数是凹函数，然后负对数函数就是凸函数；

    3.    对于一个凸函数进行仿射变换，可以理解为线性变换，结果还是凸函数；

    4.    二次函数是凸函数（二次项系数为正）；

    5.    高斯分布函数是凹函数；

    6.    多个凸函数的线性加权，如果权值是大于等于零的，那么整个加权结果函数是凸函数。

####?模型的拟合程度
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/80111526036909_.pic.jpg)
>翻译: 当观测次数趋于无穷大时，预测值和实际值之差大于某一误差值的概率趋于零的情况。

1.拟合的三种情况

看以下三张图片，这三张图片是线性回归模型 拟合的函数和训练集的关系

第一张图片拟合的函数和训练集误差较大，我们称这种情况为 欠拟合
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image015.png)
第二张图片拟合的函数和训练集误差较小，我们称这种情况为 合适拟合
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image016.png)
第三张图片拟合的函数完美的匹配训练集数据，我们称这种情况为 过拟合
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image017.png)

* 欠拟合问题，根本的原因是特征维度过少，导致拟合的函数无法满足训练集，误差较大。
欠拟合问题可以通过增加特征维度来解决。

* 过拟合问题，根本的原因则是特征维度过多，导致拟合的函数完美的经过训练集，但是对新数据的预测结果则较差。
解决过拟合问题，则有2个途径：
        1.减少特征维度; 可以人工选择保留的特征，或者模型选择算法
        2.正则化; 保留所有的特征，通过降低参数θ的值，来影响模型


####?Cp准则
by lcx

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/80131526036909_.pic.jpg)
>翻译：Mallows's Cp： 在模型选择中用于比较不同模型的表现。
n：观察样本的个数
RSS：残差平方和
d：特征个数
o：估计误差方差
2do：在估计测试误差项下对训练数据进行调整的处罚。

在统计学中，马洛斯(Colin Lingwood Mallows)提出运用Cp [1-2]  去评估一个以普通最小二乘法(Ordinary Least Square或OLS)为假设的线性回归模型的优良性，从而用于模型选取(Model Selection)。当模型中含有多个自变量(Independent Variables或Explanatory Variables)，使用Mallows’s Cp 可以为模型精选出自变量子集。Cp数值越小模型准确性越高。对于高斯线性模型(Gaussian Linear Regression)，马洛斯的Cp值被证明与赤池信息准则(Akaike Information Criterion或AIC)等效。
####?交叉熵
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/80141526036909_.pic.jpg)

设随机变量X的分布密度为p(x),q(x)是通过统计手段得到的X的近似分布，则随机变量X的交叉熵定义为：
![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image019.png)

1. 信息熵的抽象定义

熵的概念最早由统计热力学引入。信息熵是由信息论之父香农提出来的，它用于随机变量的不确定性度量，信息熵的公式：

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image020.png)

信息是用来减少随机不确定性的东西（即不确定性的减少）。

我们可以用log ( 1/P )来衡量不确定性。P是一件事情发生的概率，概率越大，不确定性越小。

可以看到信息熵的公式，其实就是log ( 1/P )的期望，就是不确定性的期望，它代表了一个系统的不确定性，信息熵越大，不确定性越大。

注意这个公式有个默认前提，就是X分布下的随机变量x彼此之间相互独立。还有log的底默认为2。

信息熵在联合概率分布的自然推广，就得到了联合熵

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image021.png)
当X, Y相互独立时，H(X, Y) = H(X) + H(Y)

当X和Y不独立时，可以用　I(X, Y) = H(X) + H(Y) - H(X, Y)　衡量两个分布的相关性，这个定义比较少用到。

2. 信息熵的实例解释

举个例子说明信息熵的作用。

比如赌马比赛，有4匹马{ A, B, C, D}，获胜概率分别为{ 1/2, 1/4, 1/8, 1/8 }，将哪一匹马获胜视为随机变量X属于 { A, B, C, D } 。

假定我们需要用尽可能少的二元问题来确定随机变量 X 的取值。

例如，问题1：A获胜了吗？　问题2：B获胜了吗？　问题3：C获胜了吗？

最后我们可以通过最多3个二元问题，来确定取值。

如果X = A，那么需要问1次（问题1：是不是A？），概率为1/2

如果X = B，那么需要问2次（问题1：是不是A？问题2：是不是B？），概率为1/4

如果X = C，那么需要问3次（问题1，问题2，问题3），概率为1/8

如果X = D，那么需要问3次（问题1，问题2，问题3），概率为1/8



那么为确定X取值的二元问题的数量为

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image022.png)
回到信息熵的定义，会发现通过之前的信息熵公式，神奇地得到了：

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image024.png)
在二进制计算机中，一个比特为0或1，其实就代表了一个二元问题的回答。也就是说，在计算机中，我们给哪一匹马夺冠这个事件进行编码，所需要的平均码长为1.75个比特。

很显然，为了尽可能减少码长，我们要给发生概率p(x)较大的事件，分配较短的码长 l(x)。这个问题深入讨论，可以得出霍夫曼编码的概念。



霍夫曼编码就是利用了这种大概率事件分配短码的思想，而且可以证明这种编码方式是最优的。我们可以证明上述现象：

为了获得信息熵为 H(x)的随机变量 x的一个样本，平均需要抛掷均匀硬币（或二元问题） h(x)次（参考猜赛马问题的案例）

信息熵是数据压缩的一个临界值（参考码长部分的案例），所以，信息熵H(X)可以看做，对X中的样本进行编码所需要的编码长度的期望值。

3. 交叉熵和KL散度

上一节说了信息熵H(X)可以看做，对X中的样本进行编码所需要的编码长度的期望值。

这里可以引申出交叉熵的理解，现在有两个分布，真实分布p和非真实分布q，我们的样本来自真实分布p。

按照真实分布p来编码样本所需的编码长度的期望为：

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image025.png)

这就是上面说的信息熵H( p )

按照不真实分布q来编码样本所需的编码长度的期望为

![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image026.png)
这就是所谓的交叉熵H( p,q )
这里引申出KL散度D(p||q) = H(p,q) - H(p) =![](https://raw.githubusercontent.com/leichenxi1996/markdownphoto/master/image027.png)
也叫做相对熵，它表示两个分布的差异，差异越大，相对熵越大。



机器学习中，我们用非真实分布q去预测真实分布p，因为真实分布p是固定的，D(p||q) = H(p,q) - H(p) 中 H(p) 固定，也就是说交叉熵H(p,q)越大，相对熵D(p||q)越大，两个分布的差异越大。

所以交叉熵用来做损失函数就是这个道理，它衡量了真实分布和预测分布的差异性。


#####3.Hamming Loss 海明距离
by lcx

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-3.jpg)

海明距离（Hamming Distance）用于需要对样本多个标签进行分类的场景。对于给定的样本i，$\hat{y}\underset{ij}{ }$是对第j个标签的预测结果，$y\underset{ij}{ }$是第j个标签的真实结果，L是标签数量，则$\hat{y}\underset{i}{ }$与$y\underset{i}{ }$间的海明距离为

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/3_1.jpg)

其中1(x)是indicator function。当预测结果与实际情况完全相符时，距离为0；当预测结果与实际情况完全不符时，距离为1；当预测结果是实际情况的真子集或真超集时，距离介于0到1之间。

我们可以通过对所有样本的预测情况求平均得到算法在测试集上的总体表现情况，当标签数量L为1时，它等于1-Accuracy，当标签数L>1时也有较好的区分度，不像准确率那么严格。

##### 9.hinge loss
by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-9.jpg)


翻译：在支持向量机中使用

在机器学习中，hinge loss常作为分类器训练时的损失函数。hinge loss用于“最大间隔”分类，特别是针对于支持向量机（SVM）。对于一个期望输出 $t=\pm 1$ 和分类分数y，预测值y的hinge loss被定义为：

$\ell(y) = \max(0, 1-t \cdot y)$

其中，y是预测值（-1到1之间），t为目标值（±1）。

其含义为，y的值在-1到1之间就可以了，并不鼓励|y|>1，即并不鼓励分类器过度自信，让某个可以正确分类的样本距离分割线的距离超过1并不会有任何奖励。从而使得分类器可以更专注整体的分类误差。


注意：这里的y分类器决策函数的“原始”输出，而不是预测的类别标签。例如，在线性SVM中，y=wx+b，(w,b)是分类超平面的参数，x是要分类的点。
可以看到，当t和y有相同的符号的时候（这意味着y的预测是正确的）并且$\left | y \right |=\pm 1$，hinge loss的结果为L(y)=0，但是当出现错误的分类是，hinge loss的L(y)与y呈线性关系（一个线性误差）。

参考资料：
[维基百科](https://en.wikipedia.org/wiki/Hinge_loss)


#####15.hypothesis space 假设空间
-by lcx
![15](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-15.jpg)
翻译：
假设空间：监督学习的目的在于学习一个由输入到输出的映射，这一映射由模型来表示。换句话说，学习的目的就在于找到最好的这样的模型。模型属于由输入空间到输出空间的映射集合，这个集合就是假设空间（hypothesis space）。假设空间的确定意味着学习范围的确定。

**假设空间与样本空间的关系**

假设空间是理论上的所有可能属性值构成的集合空间；
样本空间通常指训练数据中实际出现的所有属性值构成的集合空间。

个人理解和看法。样本空间是一个由所有可能的输入样本的特征向量所构成的空间，如果要讨论的话应该不能说他们之间有什么区别，应该是讲二者有什么关系。样本空间和假设空间的关系应该是假设空间中的一个映射将样本空间中的特征向量映射到输出空间中。
#####21.initializing weights in feedforward neural networks 在前馈神经网络中初始化权重
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-21.jpg)
翻译
1. 用小的随机数初始化
2. 常见的从正态分布中提取出的权重。
3. 偏差初始化为零或小正数

权值初始化的意义
一个好的权值初始值，有以下优点:

- 加快梯度下降的收敛速度
- 增加梯度下降到最小训练误差的几率

参数初始化的目的是为了让神经网络在训练过程中学习到有用的信息，这意味着参数梯度不应该为0。所以参数初始化要满足两个必要条件:(1)各个激活层不会出现饱和现象，比如对于sigmoid激活函数，初始化值不能太大或太小，导致陷入其饱和区。（2）各个激活值不为0，如果激活层输出为零，也就是下一层卷积层的输入为零，所以这个卷积层对权值求偏导为零，从而导致梯度为0。
权值初始化的方法
权值初始化的方法主要有：常量初始化（constant）、高斯分布初始化（gaussian）、positive_unitball初始化、均匀分布初始化（uniform）、xavier初始化、msra初始化、双线性初始化（bilinear）

- 常量初始化(constant)
把权值或者偏置初始化为一个常数，具体是什么常数，可以自己定义

- 高斯分布初始化（gaussian）
需要给定高斯函数的均值与标准差

- positive_unitball初始化
让每一个神经元的输入的权值和为 1
例如：一个神经元有100个输入，让这100个输入的权值和为1.  首先给这100个权值赋值为在（0，1）之间的均匀分布，然后，每一个权值再除以它们的和就可以啦。这么做，可以有助于防止权值初始化过大，从而防止激活函数（sigmoid函数）进入饱和区。所以，它应该比较适合simgmoid形的激活函数

- 均匀分布初始化（uniform）
将权值与偏置进行均匀分布的初始化，用min 与 max 来控制它们的的上下限，默认为（0，1）

- xavier初始化
对于权值的分布：均值为0，方差为（1 / 输入的个数） 的 均匀分布。如果我们更注重前向传播的话，我们可以选择 fan_in，即正向传播的输入个数；如果更注重后向传播的话，我们选择 fan_out, 因为在反向传播的时候，fan_out就是神经元的输入个数；如果两者都考虑的话，就选  average = (fan_in + fan_out) /2。对于ReLU激活函数来说，XavierFiller初始化也是很适合。关于该初始化方法，具体可以参考文章1、文章2，该方法假定激活函数是线性的。

- msra初始化
对于权值的分布：基于均值为0，方差为( 2/输入的个数)的高斯分布；它特别适合 ReLU激活函数，该方法主要是基于Relu函数提出的，推导过程类似于xavier，可以参考博客。

- 双线性初始化（bilinear）
常用在反卷积神经网络里的权值初始化
参考文献：
[为什么要进行权值初始化](链接：https://www.zhihu.com/question/56526007/answer/371397183)
[权值初始化的方法](https://blog.csdn.net/u013989576/article/details/76215989)
#####27.IQR 四分位数
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-27.jpg)

- 四分位距（interquartile range, IQR），又称四分差。是描述统计学中的一种方法，以确定第三四分位数和第一二分位数的区别。与方差、标准差一样，表示统计资料中各变量分散情形，但四分差更多为一种稳健统计（robust statistic）。
- 将所有数值按大小顺序排列并分成四等份,处于三个分割点位置的得分就是四分位数.最小的四分位数称为下四分位数,所有数值中,有四分之一小于下四分位数,四分之三大于下四分位数.中点位置的四分位数就是中位数.最大的四分位数称为上四分位数,所有数值中,有四分之三小于上四分位数,四分之一大于上四分位数.也有叫第25百分位数、第75百分位数的.

- 首先确定四分位数的位置：
    Q1的位置= (n+1) × 0.25
    Q2的位置= (n+1) × 0.5
    Q3的位置= (n+1) × 0.75
    n表示项数

实例1

    数据总量: 6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36
    由小到大排列的结果: 6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49
    一共11项
    Q1 的位置=（11+1） × 0.25=3， Q2 的位置=（11+1）× 0.5=6， Q3的位置=（11+1） × 0.75=9
    Q1 = 15，
    Q2 = 40，
    Q3 = 43
应用

不论Q1，Q2，Q3的变异量数数值为何，均视为一个分界点，以此将总数分成四个相等部份，可以通过Q1，Q3比较，分析其数据变量的趋势。

四分位数在统计学中的箱线图绘制方面应用也很广泛。所谓箱线图就是 由一组数据5 个特征绘制的一个箱子和两条线段的图形，这种直观的箱线图不仅能反映出一组数据的分布特征，而且还可以进行多组数据的分析比较。这五个特征值，即数据的最大值、最小值、中位数和两个四分位数。即：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/64380cd7912397ddf704b8605982b2b7d0a28750.jpg)
参考文献：
[四分位数](https://baike.baidu.com/item/四分位数/5040599?fr=aladdin)

#####39.L1范式
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-39.jpg)
L1范数是我们经常见到的一种范数，它的定义如下：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/39_1.jpg)

表示向量x中非零元素的绝对值之和。
L1范数有很多的名字，例如我们熟悉的曼哈顿距离、最小绝对误差等。使用L1范数可以度量两个向量间的差异，

由于L1范数的天然性质，对L1优化的解是一个稀疏解，因此L1范数也被叫做稀疏规则算子。通过L1可以实现特征的稀疏，去掉一些没有信息的特征

例如在对用户的电影爱好做分类的时候，用户有100个特征，可能只有十几个特征是对分类有用的，大部分特征如身高体重等可能都是无用的，利用L1范数就可以过滤掉。
参考文献
[机器学习中的范数规则化之（一）L0、L1与L2范数、核范数与规则项参数选择](https://blog.csdn.net/u012467880/article/details/52852242)

#####45.learning rate 学习率
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-45.jpg)

学习率是一个重要的超参数，它控制着我们基于损失梯度调整神经网络权值的速度，大多数优化算法（如SGD、RMSprop、Adam）对它都有涉及。学习率越小，我们沿着损失梯度下降的速度越慢。从长远来看，这种谨慎慢行的选择可能还不错，因为可以避免错过任何局部最优解，但它也意味着我们要花更多时间来收敛，尤其是如果我们处于曲线的至高点。

以下等式显示了这种关系：

新权值 = 当前权值 - 学习率 × 梯度
![](http://t11.baidu.com/it/u=2806832323,267517769&fm=173&s=04D0EC33175A51C846F975DA0000C0B2&w=640&h=347&img.JPEG)
过小（上）和过大（下）的学习率

通常，学习率是用户自己随意设的，你可以根据过去的经验或书本资料选择一个最佳值，或凭直觉估计一个合适值。这样做可行，但并非永远可行。事实上选择学习率是一件比较困难的事，下图显示了应用不同学习率后出现的各类情况：
![](http://t10.baidu.com/it/u=3891758943,1886907092&fm=173&s=09E2E9130D5AD5CE18C595DA0000C0B3&w=459&h=414&img.JPEG)
可以发现，学习率直接影响我们的模型能够以多快的速度收敛到局部最小值（也就是达到最好的精度）。一般来说，学习率越大，神经网络学习速度越快。如果学习率太小，网络很可能会陷入局部最优；但是如果太大，超过了极值，损失就会停止下降，在某一位置反复震荡。

也就是说，如果我们选择了一个合适的学习率，我们不仅可以在更短的时间内训练好模型，还可以节省各种云的花费。
参考资料：
[什么是学习率](http://baijiahao.baidu.com/s?id=1591531217345055627&wfr=spider&for=pc)

#####51.linearly separable 线性可分离
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-51.jpg)

简单的说就是如果用一个线性函数可以将两类样本完全分开，就称这些样本是“线性可分”的。
如何判断数据是线性可分的？
最简单的情况是数据向量是一维二维或者三维的，我们可以把图像画出来，直观上就能看出来。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/51_1.png)
非常简单就看出两个类的情形下X和O是不是线性可分。
但是数据向量维度一旦变得很高，我们怎么办？

答案是检查凸包（convex hull）是否相交。
什么是凸包呢？

简单说凸包就是一个凸的闭合曲线（曲面），而且它刚好包住了所有的数据。

举个例子，下图的蓝色线就是一个恰好包住所有数据的闭合凸曲线。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/51_2.png)
知道了什么是凸包，我们就能检查我们的数据是不是线性可分了。

以二维的情况为例，如果我们的数据训练集有两类：M+和M-，

当我们画出两个类的凸包，如果两者不重叠，那么两者线性可分，反之则不是线性可分。

下图就是个线性可分的情况。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/51_3.png)

什么是线性不可分

线性不可分简单来说就是你一个数据集不可以通过一个线性分类器（直线、平面）来实现分类。这样子的数据集在实际应用中是很常见的，例如：人脸图像、文本文档等。下面的几个数据都是线性不可分的：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/51_4.png)
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/51_5.png)
我们不可以使用一个直线或者一个直面把上面图像中的两类数据很好的划分。这就是线性不可分。
参考文献
[如何判断数据是线性可分](https://blog.csdn.net/u013300875/article/details/44081067)
[线性不可分的情况](https://blog.csdn.net/puqutogether/article/details/41309745)
#####57. deep learning 深度学习
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-57.jpg)
翻译
基础元素

1. 数据
2. 损失函数 例如：交叉熵
3. 优化算法 例如：Adam
4. 网咯架构 例如：稠密层
5. 测试数据
6. 评价指标 例如：准确率

概念
深度学习（Deep Learning）（也称为深度结构学习【Deep Structured Learning】、层次学习【Hierarchical Learning】或者是深度机器学习【Deep Machine Learning】）是一类算法集合，是机器学习的一个分支。它尝试为数据的高层次摘要进行建模。

用一个简单的例子来说明

假设你有两组神经元，一个是接受输入的信号，一个是发送输出的信号。当输入层接收到输入信号的时候，它将输入层做一个简单的修改并传递给下一层。在一个深度网络中，输入层与输出层之间可以有很多的层（这些层并不是由神经元组成的，但是它可以以神经元的方式理解），允许算法使用多个处理层，并可以对这些层的结果进行线性和非线性的转换。

深度学习的架构

- 生成式深度架构（Generative deep architectures），主要是用来描述具有高阶相关性的可观测数据或者是可见的对象的特征，主要用于模式分析或者是总和的目的，或者是描述这些数据与他们的类别之间的联合分布。（其实就是类似于生成模型）

- 判别式深度架构（Discriminative deep architectures），主要用于提供模式分类的判别能力，经常用来描述在可见数据条件下物体的后验类别的概率。（类似于判别模型）

- 混合深度架构（Hybrid deep architectures），目标是分类，但是和生成结构混合在一起了。比如以正在或者优化的方式引入生成模型的结果，或者使用判别标注来学习生成模型的参数。

实际中对应的模型的例子就是深度前馈网络，卷积网络和递归神经网络（Deep feed-forward networks, Convolution networks and Recurrent Networks)。

- 深度前馈网络（Deep feed-forward networks）
深度前馈网络也叫做前馈神经网络，或者是多层感知机（Multilayer Perceptrons，MLPs），是深度学习模型中的精粹。

- 前馈网络的目标是近似某些函数。例如，对于一个分类器，y=f(x)来说，它将一个输入值x变成对应的类别y。前馈网络就是定义一个映射y=f(x;θ)，并学习出参数θ使得产生最好的函数近似。

    简而言之，神经网络可以定义成输入层，隐含层和输出层。其中，输入层接受数据，隐含层处理数据，输出层则输出最终结果。这个信息流就是接受x，通过处理函数f，在达到输出y。这个模型并没有任何的反馈连接，因此被称为前馈网络。

- 卷积神经网络（Convolution Neural Networks）
在机器学习中，卷积神经网络（简称CNN或者ConvNet）是一种前馈神经网络，它的神经元的连接是启发于动物视觉皮层。单个皮质神经元可以对某个有限空间区域的刺激作出反应。这个有限空间可以称为接受域。不同的神经元的接受域可以重叠，从组成了所有的可见区域。那么，一个神经元对某个接受域内的刺激作出反应，在数学上可以使用卷积操作来近似。也就是说，卷积神经网络是受到生物处理的启发，设计使用最少的预处理的多层感知机的变体。
    卷积神经网络在图像和视频识别、推荐系统以及自然语言处理中都有广泛的运用。

    LeNet是早期推动深度学习发展的卷积神经网络之一。这是Yann LeCun从1988年以来进行的许多词的成功迭代后得到的开创性工作，称之为LeNet5。在当时，LeNet架构主要用来进行字符识别的工作，如读取邮编，数字等。卷积神经网络主要包含四块：卷积层（Convolutional Layer）、激活函数（Activation Function）、池化层（Pooling Layer）、全连接层（Fully Connected Layer）。

  - 卷积层（Convolutional Layer）
    卷积层是基于单词“卷积（Convolution）”而来，这是一种数学上的操作，它是对两个变量f*g进行操作产生第三个变量。它和互相关（cross-correlation）很像。卷积层的输入是一个m×m×r图像，其中m是图像的高度和宽度，r是通道的数量，例如，一个RGB图像的通道是3，即r=3。卷积层有k个滤波器【filters】（或者称之为核【kernel】），其大小是n×n×q，这里的n是比图像维度小的一个数值，q既可以等于通道数量，也可以小于通道数量，具体根据不同的滤波器来定。滤波器的大小导致了

  - 激活函数（Activation Function）
    为了实现复杂的映射函数，我们需要使用激活函数。它可以带来非线性的结果，而非线性可以使得我们很好的拟合各种函数。同时，激活函数对于压缩来自神经元的无界线性加权和也是重要的。 激活函数很重要，它可以避免我们把大的数值在高层次处理中进行累加。激活函数有很多，常用的有sigmoid，tanh和ReLU。

  - 池化层（Pooling Layer）
    池化是一个基于样本的离散化过程。其目的上降低输入表示的采样（这里的输入可以是图像，隐层的输出等），减少它们的维度，并允许我们假设特征已经被包含在了子区域中。
    这部分的作用是通过提供一种抽象的形式表示来帮助过拟合表示。同样的，它也通过减少了参数的数量降低了计算的复杂度并为内部的表示提供一个基本的不变性的转换。
目前最常用的池化技术有Max-Pooling、Min-Pooling和Average-Pooling。

  - 连接层（Fully Connected Layer）
    “全连接”的意思是指先前的层里面的所有的神经元都与后一个层里面的所有的神经元相连。全连接层是一种传统的多层感知机，在输出层，它使用softmax激活函数或者其他激活函数。

  - 递归神经网络（Recurrent Neural Networks）
    在传统的神经网络中，我们假设所有的输入之间相互独立。但是对于很多任务来说，这并不是一个好的主意。如果你想知道一个句子中下一个单词是什么，你最好知道之前的单词是什么。RNN之所以叫RNN就是它对一个序列中所有的元素都执行相同的任务，所有的输出都依赖于先前的计算。另一种思考RNN的方式是它会记住所有之前的计算的信息。

应用
在实际应用中，很多问题都可以通过深度学习解决。那么，我们举一些例子：

- 黑白图像的着色

深度学习可以用来根据对象及其情景来为图片上色，而且结果很像人类的着色结果。这种解决方案使用了很大的卷积神经网络和有监督的层来重新创造颜色。

- 机器翻译

深度学习可以对未经处理的语言序列进行翻译，它使得算法可以学习单词之间的依赖关系，并将其映射到一种新的语言中。大规模的LSTM的RNN网络可以用来做这种处理。

- 图像中的对象分类与检测

这种任务需要将图像分成之前我们所知道的某一种类别中。目前这类任务最好的结果是使用超大规模的卷积神经网络实现的。突破性的进展是Alex Krizhevsky等人在ImageNet比赛中使用的AlexNet模型。

- 自动产生手写体

这种任务是先给定一些手写的文字，然后尝试生成新的类似的手写的结果。首先是人用笔在纸上手写一些文字，然后根据写字的笔迹作为语料来训练模型，并最终学习产生新的内容。

- 自动玩游戏

这项任务是根据电脑屏幕的图像，来决定如何玩游戏。这种很难的任务是深度强化模型的研究领域，主要的突破是DeepMind团队的成果。

- 聊天机器人

一种基于sequence to sequence的模型来创造一个聊天机器人，用以回答某些问题。它是根据大量的实际的会话数据集产生的。
参考文献
[深度学习入门](https://www.zhihu.com/question/26006703/answer/126777449)
[深度学习必须理解的25个概念](https://blog.csdn.net/pangjiuzala/article/details/72630166)
[深度学习应用](https://zhuanlan.zhihu.com/p/25482889)
[深度学习基本架构以及原理](http://m.elecfans.com/article/579639.html)
#####63.mean absolute error 绝对平均误差
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-63.jpg)

公式：
${\rm MAE}(y, \hat{y})=\frac{1}{n_{\rm samples}}\sum\limits_{i=1}^{n_{\rm samples}}|y_i-\hat{y}_i|$
对同一物理量进行多次测量时，各次测量值及其绝对误差不会相同，我们将各次测量的绝对误差取绝对值后再求平均值，并称其为平均绝对误差

#####75.model identifiability 模型可识别性
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-75.jpg)

翻译：如果可以使用数据来查找单个最好的参数集，那么模型就是可识别的。

为了识别，讨论一个参数θ（可以是矢量），其范围在参数空间$\Theta $以及由θ索引的分布族,我们通常会写${fθ|θ∈Θ}$

例如,θ可能是θ=β和f可能是
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/75_1.jpg)
这意味着$Θ = （0 ，∞ ）$。为了使模型可以被识别，映射θ的变换到fθ应该是一对一的。给定一个模型，检查这个最直接的方法是从等式f开始$θ1= fθ2$，（这种平等应该适用于（几乎）所有x在支持中）并尝试使用代数（或其他参数）来表明，只有这样一个方程意味着，事实上，θ1= θ2。

如果你通过这个计划获得成功，那么你的模型是可识别的;如果不是这样，那么你的模型不可识别，或者你需要找到另一个论点。无论如何，道理是一样的：在一个可识别的模型中，两个不同的参数（可能是向量）不可能产生相同的似然函数。

就是说你待估计的参数能不能由现有方程解出来。一个解就是恰好识别，无解就是不可识别，无穷解就是过度识别。

#####87.normal distribution 正态分布
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-87.jpg)

正态分布（Normal distribution），也称“常态分布”，又名高斯分布（Gaussian distribution），最早由A.棣莫弗在求二项分布的渐近公式中得到。

正态曲线呈钟型，两头低，中间高，左右对称因其曲线呈钟形，因此人们又经常称之为钟形曲线。

若随机变量X服从一个数学期望为μ、方差为σ^2 的正态分布，记为N(μ，σ^2 )。其概率密度函数为正态分布的期望值μ决定了其位置，其标准差σ决定了分布的幅度。当μ = 0,σ = 1时的正态分布是标准正态分布。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/87-1.jpg)

#####99.odds 比值
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-99.jpg)


 odds: 称为几率、比值、比数，是指某事件发生的可能性(概率)与不发生的可能性（概率）之比。用p表示事件发生的概率，则：odds = p/(1-p)。


#####111.parameter sharing 参数共享
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-111.jpg)

翻译：参数共享是当多个模型或部分模型共享参数时。例如，卷积神经网络在不同的部件之间共享参数。
这允许CNN在图像的任何地方识别模式，并减少需要存储的参数数量。

简单从共享的角度来说：权重共享即filter的值共享

- input是一维的情况，输入W = 5，padding = 1
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/111-3.jpg)

    此时的filter是 1 0 -1 最右上角图，即F = 3。左边stride=1，右边stride = 2。整个滤波（卷积）的过程中filter从input最左端滑到最右端，值一直保持不变更明显的一张图
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/111-2.jpg)
由固定的红色、紫色、绿色三种值组成。前一层（白色圆）总是通过相同的filter值得到上一层

- input是三维的情况，
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/111-1.jpg)
此时，每个filter需要F*F*D1个权重值，总共K个filter，需要F*F*D1*K权重值。和一维一样，整个滑动过程中filter W0和W1值保持不变，可称作权值共享。而且，补充一句，对于三维的input，权值只是在input的每个depth slice上共享的

参考文献：
[链接：https://www.zhihu.com/question/47158818/answer/128939450]()


#####117. correlation 相关系数
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-117.jpg)

公式定义为： 两个连续变量(X,Y)的pearson相关性系数(Px,y)等于它们之间的协方差cov(X,Y)除以它们各自标准差的乘积(σX,σY)。系数的取值总是在-1.0到1.0之间，接近0的变量被成为无相关性，接近1或者-1被称为具有强相关性。

皮尔森相关系数是衡量线性关联性的程度，p的一个几何解释是其代表两个变量的取值根据均值集中后构成的向量之间夹角的余弦。

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/117-1.png)

#####123. precision 精确率
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-123.jpg)

翻译：精确率是分类器不把真正的负面观察标记为正面的能力

精确率是针对我们预测结果而言的，它表示的是预测为正的样本中有多少是真正的正样本。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/123-1.jpg)

#####129.PMF 概率质量函数
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-129.jpg)

在概率论中，概率质量函数 (Probability Mass Function，PMF)是离散随机变量在各特定取值上的概率。

概率质量函数和概率密度函数不同之处在于：概率密度函数是对连续随机变量定义的，本身不是概率，只有对连续随机变量的取值进行积分后才是概率。

假设有一元随机变量。

- 如果是连续型随机变量，那么可以定义它的概率密度函数（probability density function, PDF），![](https://www.zhihu.com/equation?tex=f_X%28x%29)有时简称为密度函数。
![](https://www.zhihu.com/equation?tex=%5CPr%5Cleft%28a+%5Cleq++X+%5Cleq+b%5Cright%29+%3D%5Cint_%7Ba%7D%5E%7Bb%7D+f_X%28x%29+dx)

    我们用PDF在某一区间上的积分来刻画随机变量落在这个区间中的概率，即：
![](https://www.zhihu.com/equation?tex=%5CPr%5Cleft%28a+%5Cleq++X+%5Cleq+b%5Cright%29+%3D%5Cint_%7Ba%7D%5E%7Bb%7D+f_X%28x%29+dx)

- 如果是离散型随机变量，那么可以定义它的概率质量函数（probability mass function, PMF）![](https://www.zhihu.com/equation?tex=f_X%28x%29)。

    与连续型随机变量不同，这里的PMF其实就是高中所学的离散型随机变量的分布律，即
![](https://www.zhihu.com/equation?tex=f_X%28x%29%3D%5CPr%5Cleft%28+X%3Dx+%5Cright%29+)

    比如对于掷一枚均匀硬币，如果正面令，如果反面令，那么它的PMF就是
![](https://www.zhihu.com/equation?tex=f_X%5Cleft%28+x+%5Cright%29+%3D%5Cbegin%7Bcases%7D%0A+%26%5Cfrac%7B1%7D%7B2%7D+%5Ctext%7B+if+%7D+x%5Cin%5Cleft+%5C%7B+0%2C1+%5Cright+%5C%7D+%5C%5C+%0A+%26+0%5Ctext%7B+if+%7D+x%5Cnotin%5Cleft+%5C%7B+0%2C1+%5Cright+%5C%7D%0A%5Cend%7Bcases%7D)


参考文献
[https://www.zhihu.com/question/36853661/answer/69775009]()

#####135.Boosting
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-135.jpg)

翻译：一种集成学习策略，训练一系列弱模型，每个模型都试图正确地预测先前模型错误的观测结果

提升算法(Boosting)是常用的有效的统计学习算法，属于迭代算法，它通过不断地使用一个弱学习器弥补前一个弱学习器的“不足”的过程，来串行地构造一个较强的学习器，这个强学习器能够使目标函数值足够小。从优化的角度分析，与一般的在参数空间搜索最优解的学习算法（如神经网络）类似，

Boosting也是一个迭代搜索，且最优的算法，不同的是，它的搜索空间是学习器空间，或说函数空间（Function space），它的搜索方向是构造不断完善的强学习器，以达到目标函数（或说误差函数）足够小的目的。
基本思想
基本思想：
1) 先赋予每个训练样本相同的概率。
2) 然后进行T次迭代，每次迭代后，对分类错误的样本加大权重(重采样)，使得在下一次的迭代中更加关注这些样本。
![](http://blog.chinaunix.net/attachment/201203/12/8695538_1331555368jWSr.jpg)
示例：
![](http://blog.chinaunix.net/attachment/201203/12/8695538_1331555414I2if.jpg)
主要过程

为说明Boosting的主要过程，下面举一个简化的例子。

假设训练数据集为(x1,y1),(x2,y2),...,(xn,yn)，我们的任务是寻找使这个回归问题的均方误差最小的模型F(x)。

如果已经有一个初始的模型f,且f(x1)=0.8，但y1=0.9，f(x2)=1.4，但y2=1.3… 显然f是不完美的，我们可以采用不断完善f的方式,如不断在f的基础上增加模型（如决策树）h，即：f(x)←f(x)+h(x),使f趋于F.

我们希望：
 ![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/135-1.jpg)
即
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/135-2.jpg)
然而恰好满足上式的h
可能不存在，但我们总可以找到使残差yi−f(xi)变小的h.

上述过程等价于拟合如下数据集：

(x1,y1−f(x1)),(x2,y2−f(x2)),...,(xn,yn−f(xn))

上述一次叠加h的过程就是Boosting的一次迭代。要使f
足够接近F，一般需要多次迭代。

参考文献
[boosting原理](https://blog.csdn.net/laiqun_ai/article/details/46761391)

#####141.RSS 剩余平方和
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-141.jpg)

翻译：对错误进行平方将更严重地惩罚一些较大的错误，即使错误的总和是相同的

- 它表明除x对y的线性变化之外的一切因素(包括x对y的非线性影响及测量误差等)对y的离差的影响。
- 残差平方和是用连续曲线近似地刻画或比拟平面上离散点组，以表示坐标之间函数关系的一种数据处理方法。用解析表达式逼近离散数据的一种方法。
- 为了明确解释变量和随机误差各产生的效应是多少，统计学上把数据点与它在回归直线上相应位置的差异称为残差，把每个残差平方之后加起来 称为残差平方和，它表示随机误差的效应。
- 每一点y的估计值与实际值之差的平方之和称为残差平方和,而y的实际值和平均值的差的平方之和称为总平方和；简单来说,一组数据的残差平方和越小,其拟合程度越好。

参考文献
[rss](https://zhidao.baidu.com/question/506219148.html)

#####147. saturation 饱和
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-147.jpg)

翻译：当一个函数的输出对输入非常不敏感的时候，即饱和。例如：sigmoid

当sigmoid和tanh等激活函数值接近其边界值（对sigmoid就是0或者1）的时候，会导致算法在反向传播时梯度消失。

例如在反向传播中，梯度将会与整个损失函数关于该门单元输出的梯度相乘，因此，如果局部梯度非常小，那么相乘的结果也会趋近于零，这就会有效的杀死梯度，几乎就没有梯度信号通过神经元传到权重再到数据了。
![](https://pic3.zhimg.com/80/677187e96671a4cac9c95352743b3806_hd.jpg)

就是指梯度接近于0（例如sigmoid函数的值接近于0或1时，函数曲线平缓，梯度接近于0）的状态，这会导致采用梯度下降法求最值时，速度偏缓。

参考文献：
[深度学习中saturation是什么意思](https://www.zhihu.com/question/48010350/answer/109446932)

#####153. Simpson‘s paradox 辛普森悖论
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-153.jpg)

翻译
当在群体分离时，出现了一种趋势，但当群体合并时，就会出现另一种趋势。

简介

辛普森悖论是一种统计现象，实验群体由具有不同统计特性的子群体组成，观察到的现象是总体水平可能与单个子群体的水平不相关。换句话说，辛普森悖论是在一个数据集中的变量被分组之后，他们之间的相关性可能会发生改变。

辛普森悖论在数据集方面看上去广泛，而且没有被分解成有意义的片段。辛普森悖论是研究中被忽略的“混淆变量”结果。混淆变量本质上是一个与核心研究无关的变量，它随着自变量的改变而改变。

为了避免辛普森悖论的出现，就需要斟酌各分组的权重，并乘以一定的系数去消除以分组数据基数差异而造成的影响。同时必需了解清楚情况，是否存在潜在因素，综合考虑。

经典实例

(以下内容取材自维基百科与科普写作奖佳作奖作者林守德的向理性与直觉挑战的顽皮精灵-综观诡谲的悖论等文)

“校长，不好了，有很多男生在校门口抗议，他们说今年研究所女生录取率42%是男生21%的两倍，我们学校遴选学生有性别歧视”，校长满脸疑惑的问秘书：“我不是特别交代，今年要尽量提升男生录取率以免落人口实吗？”

秘书赶紧回答说：“确实有交代下去，我刚刚也查过，的确是有注意到，今年商学院录取率是男性75%，女性只有49%；而法学院录取率是男性10%，女性为5%。二个学院都是男生录取率比较高，校长这是我作的调查报告。”

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/153_1.jpg)

“秘书，你知道为什么个别录取率男皆大于女，但是总体录取率男却远小于女吗？”
此例这就是统计上著名的辛普森悖论(Simpson's Paradox)

上面例子说明，简单的将分组资料相加汇总，是不一定能反映真实情况的。就上述例子录取率与性别来说，导致辛普森悖论有两个前提。

(1) 两个分组的录取率相差很大，就是说法学院录取率9.2%很低，而商学院53.3%却很高，另一方面，两种性别的申请者分布比重却相反，女生偏爱申请商学院，故商学院女生申请比率占83.3%，相反男生偏爱申请法学院，因此法学院女生申请比率只占0.833%。结果在数量上来说，录取率低的法学院，因为女生申请为数少，所以不录取的女生相对很少。而录取率很高的商学院虽然录取了很多男生，但是申请者却不多。使得最后汇总的时候，女生在数量上反而占优势。

(2) 性别并非是录取率高低的唯一因素，甚至可能是毫无影响的，至于在法商学院中出现的比率差可能是属于随机事件，又或者是其他因素作用，譬如学生入学成绩却刚好出现这种录取比例，使人牵强地误认为这是由性别差异而造成的。

解决方法

数据分析中为了避免辛普森悖论出现，就需要斟酌个别分组的权重，以一定的系数去消除以分组资料基数差异所造成的影响，同时必需了解该情境是否存在其他潜在要因而综合考虑。

参考文献
[辛普森悖论](https://baike.baidu.com/item/辛普森悖论/4475862?fr=aladdin)

#####159.sources of uncertainty 不确定性的来源
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-159.jpg)
翻译:
1. 宇宙固有的随机性，例如：量子力学
2. 不能完全观察现象，即使它是确定性的，例如：观察犯罪
3. 无法完美地模拟一种现象，例如：预测犯罪的模型是简单化的。

概率论是用于表示不确定性声明的数学框架。它不仅提供了量化不确定性的方法，也提供了用于导出新的不确定性的方法。计算机科学的许多分支处理的实体大部分都是完全确定。然而机器学习中经常要处理随机量，所以经常要用到概率论的知识。

不确定性有三种可能的来源：

1. 被建模系统内在的随机性。
例如，大多数量子力学的解释，都将亚原子粒子的动力学描述为概率的。我们还可以创建一些我们假设具有随机动态的理论情境，例如一个假想的纸牌游戏，在这个游戏中我们假设纸牌被真正混洗成了随机顺序。

2. 不完全观测。
即使是确定的系统，当我们不能观测到所有驱动系统行为的变量时，该系统也会呈现随机性。例如，在Monty Hall问题中，一个游戏节目的参与者被要求在三个门之间选择，并且会赢得放置在选中门后的奖品。其中两扇门通向山羊，第三扇门通向一辆汽车。选手的每个选择所导致的结果是确定的，但是站在选手的角度，结果是不确定的。

3. 不完全建模。
当我们使用一些必须舍弃某些观测信息的模型时，舍弃的信息会导致模型的预测出现不确定性。例如，假设我们制作了一个机器人，它可以准确地观察周围每一个对象的位置。在对这些对象将来的位置进行预测时，如果机器人采用的是离散化的空间，那么离散化的方法将使得机器人无法确定对象们的精确位置：因为每个对象都可能处于它被观测到的离散单元的任何一个角落。

概率可以被看作是用于处理不确定性的逻辑扩展。逻辑提供了一套形式化的规则，可以在给定某些命题是真或假的假设下，判断另外一些命题是真的还是假的。概率论提供了一套形式化的规则，可以在给定一些命题的真假后，计算其他命题的真假。
参考文献：
[不确定性的来源](http://blog.sina.com.cn/s/blog_182ec9dc30102ybjw.html)

#####165.standardization 标准化
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-165.jpg)

翻译
标准化是一种常见的扩展方法。Xi'表示每个值离均值的标准差数，它将特征值重新设为0的均值和单位方差

数据的标准化是将数据按比例缩放，使之落入一个小的特定区间。由于信用指标体系的各个指标度量单位是不同的，为了能够将指标参与评价计算，需要对指标进行规范化处理，通过函数变换将其数值映射到某个数值区间。

在机器学习中，我们可以处理各种类型的数据，例如音频信号和图像数据的像素值，这些数据可以包含多个维度。特征标准化使得数据中每个特征的值具有零均值（当减去分子中的均值时）和单位方差。这种方法被广泛用于许多机器学习算法（例如支持向量机，逻辑回归和人工神经网络）中的归一化。

一般的计算方法是确定分布均值和标准差为每个功能。接下来我们从每个特征中减去平均值。然后我们将每个特征的值（均值已经减去）除以其标准偏差。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/165_1.jpg)

X 是原始的特征向量，$\overline{x}$是该特征向量的平均值，并且$\delta $是它的标准偏差。
参考文献：
[标准化数据](https://en.wikipedia.org/wiki/Feature_scaling)

#####171. strategies for highly imbalanced classes 高度不平衡类的策略
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-171.jpg)

1. 收集更多的数据
2. 选择适合于不平衡类(如精度或撤销)的损失函数
3. 增大权重
4. 下采样和上采样

不平衡数据

什么是不平衡数据呢？顾名思义即我们的数据集样本类别极不均衡，以二分类问题为例，假设我们的数据集是$S$，数据集中的多数类为$S_maj$，少数类为$S_min$，通常情况下把多数类样本的比例为$100:1$,$1000:1$，甚至是$10000:1$这种情况下为不平衡数据，不平衡数据的学习即需要在如此分布不均匀的数据集中学习到有用的信息。

为什么要不平衡学习

传统的学习方法以降低总体分类精度为目标，将所有样本一视同仁，同等对待，如下图1所示，造成了分类器在多数类的分类精度较高而在少数类的分类精度很低。机器学习模型都有一个待优化的损失函数，以我们最常用最简单的二元分类器逻辑回归为例，其损失函数如下公式1所示，逻辑回归以优化总体的精度为目标，不同类别的误分类情况产生的误差是相同的，考虑一个$500:1$的数据集，即使把所有样本都预测为多数类其精度也能达到$500/501$之高，很显然这并不是一个很好的学习效果，因此传统的学习算法在不平衡数据集中具有较大的局限性。

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/171_1.png)

不平衡学习的方法

既然传统的学习算法在不平衡数据中具有较大的局限性，那么针对不平衡数据集又有怎样的解决方案呢？解决方法主要分为两个方面，第一种方案主要从数据的角度出发，主要方法为抽样，既然我们的样本是不平衡的，那么可以通过某种策略进行抽样，从而让我们的数据相对均衡一些；第二种方案从算法的角度出发，考虑不同误分类情况代价的差异性对算法进行优化，使得我们的算法在不平衡数据下也能有较好的效果。

1. 收集更多的少数类数据

2. 设计适用于不平衡数据集的模型
大部分方法都集中在数据上，并将模型保持为固定的组件。但事实上，如果设计的模型适用于不平衡数据，则不需要重新采样数据，著名的XGBoost已经是一个很好的起点，因此设计一个适用于不平衡数据集的模型也是很有意义的。
通过设计一个代价函数来惩罚稀有类别的错误分类而不是分类丰富类别，可以设计出许多自然泛化为稀有类别的模型。例如，调整SVM以惩罚稀有类别的错误分类。
![](https://static.leiphone.com/uploads/new/article/740_740/201706/59410724276fb.png?imageMogr2/format/jpg/quality/90)

3. 加权
除了采样和生成新数据等方法，我们还可以通过加权的方式来解决数据不平衡问题，即对不同类别分错的代价不同，如下图：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/171_2.jpg)
横向是真实分类情况，纵向是预测分类情况，C(i,j)是把真实类别为j的样本预测为i时的损失，我们需要根据实际情况来设定它的值。
这种方法的难点在于设置合理的权重，实际应用中一般让各个分类间的加权损失值近似相等。当然这并不是通用法则，还是需要具体问题具体分析。

4. 上采样和下采样
采样分为上采样（Oversampling）和下采样（Undersampling），上采样是把小众类复制多份，下采样是从大众类中剔除一些样本，或者说只从大众类中选取部分样本。

    随机采样最大的优点是简单，但缺点也很明显。上采样后的数据集中会反复出现一些样本，训练出来的模型会有一定的过拟合；而下采样的缺点显而易见，那就是最终的训练集丢失了数据，模型只学到了总体模式的一部分。

    上采样会把小众样本复制多份，一个点会在高维空间中反复出现，这会导致一个问题，那就是运气好就能分对很多点，否则分错很多点。为了解决这一问题，可以在每次生成新数据点时加入轻微的随机扰动，经验表明这种做法非常有效。

    因为下采样会丢失信息，如何减少信息的损失呢？
  - 第一种方法叫做EasyEnsemble，利用模型融合的方法（Ensemble）：多次下采样（放回采样，这样产生的训练集才相互独立）产生多个不同的训练集，进而训练多个不同的分类器，通过组合多个分类器的结果得到最终的结果
  - 第二种方法叫做BalanceCascade，利用增量训练的思想（Boosting）：先通过一次下采样产生训练集，训练一个分类器，对于那些分类正确的大众样本不放回，然后对这个更小的大众样本下采样产生训练集，训练第二个分类器，以此类推，最终组合所有分类器的结果得到最终结果。第三种方法是利用KNN试图挑选那些最具代表性的大众样本，叫做NearMiss，这类方法计算量很大。

参考文献
[机器学习中的数据不平衡解决方案大全](链接：https://www.jianshu.com/p/3e8b9f2764c8)
[不平衡数据下的机器学习方法简介](https://www.jianshu.com/p/3e8b9f2764c8)

#####183.tests、training、validation sets 测试、训练、验证集
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-183.jpg)

翻译：
训练集：这部分数据被用来寻找最小化损失函数的合适的权重
验证集：用于调优学习算法的超参数的数据
测试集：用于评价模型普遍性的数据

#####195. TSS total sum of squares 完全平方和
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-195.jpg)

在统计数据分析中，总平方和（TSS或SST）是作为呈现这些分析结果的标准方式的一部分出现的量。它被定义为所有观测数据中每个观测值与总平均值的平方差的总和。


![](https://wikimedia.org/api/rest_v1/media/math/render/svg/24f49fd012d7208436fc502fdb1f0065605951e6)

 $\overline{y}$是平均值。

#####201. chain rule of calculus 链式法则的微积分
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-201.jpg)

翻译：用于有效地计算反向传播中的梯度。如果$y=g(x)$，$z=f(g(x))$，则$f(y)$:
$\frac{dz}{dx}=\frac{dz}{dy}\cdot \frac{dy}{dx}$
反向传播通常在张量上使用链式法则，但概念本质上是相同的

链式法则是微积分中的求导法则，用于求一个复合函数的导数，是在微积分的求导运算中一种常用的方法。复合函数的导数将是构成复合这有限个函数在相应点的 导数的乘积，就像锁链一样一环套一环，故称链式法则。

#####207.vanishing gradient problem 梯度问题消失
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-207.jpg)

翻译：当损失函数相对于网络早期层的参数的梯度很小时。导致学习缓慢，而且由于许多梯度是微小的，它们对学习没有多大贡献，并可能导致糟糕的表现
问题引入
随着隐藏层数目的增加，分类准确率反而下降了。为什么？
先看一组试验数据，当神经网络在训练过程中, 随epoch增加时各隐藏层的学习率变化。
两个隐藏层：[784,30,30,10]
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_1.png)

三个隐藏层：[784,30,30,30,10]
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_2.png)
四个隐藏层：[784,30,30,30,30,10]
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_3.png)

可以看到：前面的隐藏层的学习速度要低于后面的隐藏层。

这种现象普遍存在于神经网络之中， 叫做消失的梯度问题（vanishing gradient problem）。

另外一种情况是内层的梯度被外层大很多，叫做激增的梯度问题（exploding gradient problem）。

更加一般地说，在深度神经网络中的梯度是不稳定的，在前面的层中或会消失，或会激增。这种不稳定性才是深度神经网络中基于梯度学习的根本问题。
产生消失的梯度问题的原因

先看一个极简单的深度神经网络：每一层都只有一个单一的神经元。如下图：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_4.jpeg)
代价函数C对偏置b1的偏导数的结果计算如下：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_5.jpeg)
先看一下sigmoid 函数导数的图像：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_6.jpeg)

该导数在σ′(0) = 1/4时达到最高。现在，如果我们使用标准方法来初始化网络中的权重，那么会使用一个均值为0 标准差为1 的高斯分布。因此所有的权重通常会满足|wj|<1。从而有wjσ′(zj) < 1/4。

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/207_7.jpeg)
这其实就是消失的梯度出现的本质原因了。

可以考虑将权重初始化大一点的值，但这可能又会造成激增的梯度问题。

根本的问题其实并非是消失的梯度问题或者激增的梯度问题，而是在前面的层上的梯度是来自后面的层上项的乘积。所以神经网络非常不稳定。唯一可能的情况是以上的连续乘积刚好平衡大约等于1，但是这种几率非常小。

所以只要是sigmoid函数的神经网络都会造成梯度更新的时候极其不稳定，产生梯度消失或者激增问题。
解决梯度消失问题
使用ReLU。
使用ReL 函数时：gradient = 0 (if x < 0), gradient = 1 (x > 0)。不会产生梯度消失问题。

参考文献
[梯度消失问题](https://www.cnblogs.com/tsiangleo/p/6151560.html)

#####213.Visualizing RSS 可视化Rss
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-213.jpg)

#####219. word 2 vec
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-219.jpg)

翻译：使用一个浅的神经网络将单词映射到一个向量空间，在这个空间中，具有类似上下文的单词有紧密的向量。

- 从需求入门

给你一个川普的词，你会联想到哪些？正常的话，应该是美国、大选、希拉里、奥巴马；也就是相似词语的选取了。对于相识词的选取，算法非常的多。也有许多人用了很简单的办法就能求得两样东西的相似性，比如购物车里物品的相似度，最简单的办法就是看看同时买了这样东西的用户还同时买了什么，用简单的数据结构就很容易实现这样的一个算法。这种算法很简单也很方便，但就是这种简单而使他忽略了很多的问题。例如时间顺序，下面会有提到。

还是回归到相识度的问题。归结到数学问题上，最经常用的是把每个词都归结到一个坐标系下，再用距离公式（如：皮尔逊公式）可方便的求出各个词语之间的相识度。

这也是word2vec的方法，word2vec 通过训练，可以把对文本内容的处理简化为 K 维向量空间中的向量运算，而向量空间上的相似度可以用来表示文本语义上的相似度。
如图，下面是有五维向量空间的单词：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/219_1.png)

算法的关键步骤就是如何求出词语的向量空间。

- word2vec算法介绍

word2vec是2013年Google中开源的一款工具。2013年神经网络的各种算法都已经相当的成熟了，word2vec核心是神经网络的方法，采用 CBOW（Continuous Bag-Of-Words，即连续的词袋模型）和 Skip-Gram 两种模型，将词语映像到同一坐标系，得出数值向量的高效工具。

一般来说算法采用神经网络的话，要注意他的输入和输出。因为使用神经网络进行训练需要有输入和输出，输入通过神经网络后，通过和输入对比，进行神经网络的重新调整，达到训练网络的目的。抓住输入输出就能够很好的理解神经网络的算法过程。

语言模型采用神经网络，就要判断什么东西要作为输入，什么东西要作为输出。这是算法可以创新的地方，语言模型有许多种，大部分的原理也是采用根据上下文，来推测这个词的概率。

word2vec输入输出也算是鬼斧神功，算法跟哈夫曼树有关系。哈夫曼树可以比较准确的表达这边文章的结构。

a,b,c,d分别表示不同词，并附加找个词出现的频率，这些词就能有自己的路径和编码。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/219_2.png)


关于哈夫曼树就不仔细详细说明了，他是一种压缩算法，能很好的保持一篇文章的特性。

训练的过程是，把每一段落取出来，每个词都通过哈夫曼树对应的路径和编码。编码是(0和1)，作为神经网络的输出，每个路径初始化一个给定维数的向量，跟自己段落中的每个词作为输入，进行反向的迭代，就可以训练出参数。

这就是算法的整个过程。

- 应用

word2vec是根据文章中每个词的上下关系，把每个词的关系映射到同一坐标系下，构成了一个大矩阵，矩阵下反映了每个词的关系。这些词的关系是通过上下文相关得出来的，它具有前后序列性，而Word2vec同时采用了哈夫曼的压缩算法，对是一些热门词进行了很好的降权处理。因此他在做一些相似词，或者词语的扩展都有很好的效果。

这种相识性还可以用在，物品的推荐上，根据用户购买物品的顺序，把每个物品当成一个单词，相当于一门外语了，谁也看不懂而已，但里面放映了上下文的关系，这个是很重要的，也是我们一开头那种普通算法无法做到的，同时对一些热门的物品自然有降权的处理，非常的方便。

word2vec自然规避了两大问题：词语的次序和热门词语的降权处理。

- 参考文献
[机器学习系列-word2vec篇](https://www.jianshu.com/p/3cda276079c7)

#####237.cost and loss function 成本和损失函数
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-237.jpg)
>翻译：成本函数和损失函数是相同的，他们都是目标函数，我们训练模型的目的就是为了使目标函数最小化，例如：交叉熵
- 概念
机器学习模型关于单个样本的预测值与真实值的差称为损失。损失越小，模型越好，如果预测值与真实值相等，就是没有损失。
用于计算损失的函数称为损失函数。模型每一次预测的好坏用损失函数来度量。

- 常用的损失函数
有以下几种（引用自李航的《统计学习方法》）

1. 0-1损失函数
![](https://img-blog.csdn.net/20180201113727804?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamllbWluZzIwMDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
二类分类任务中，预测值与真实值不同，就是预测错误，则损失是1；
预测值与真实值相等，就是预测正确，损失是 0，就是没有损失。


2. 平方损失函数
![](https://img-blog.csdn.net/20180201113822313?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamllbWluZzIwMDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
预测值与真实值的差的平方。预测误差越大，损失越大。

3. 绝对损失函数
![](https://img-blog.csdn.net/20180201113836326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamllbWluZzIwMDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
预测值与真实值的差的绝对值。绝对值不方便计算，一般不常用。

4. 对数损失函数
![](https://img-blog.csdn.net/20180201113848114?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamllbWluZzIwMDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
对于预测值是概率的情况，取对数损失。因为概率范围[0, 1]，所以对数值是(-∞, 0) ，为了让损失 > 0 所以取对数的负值。上面的公式里面有个负号。
![](https://img-blog.csdn.net/20180201113858644?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvamllbWluZzIwMDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

####243. dataset augmentation 数据集扩充
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-243.jpg)
翻译：
1. 常用于图像识别
2. 在计算机视觉中，把图像通过旋转、缩放、移动等产生噪音等等。在其他问题中也采用了其他的注入噪声的方法。
3. 可以大大减少泛化误差。

- 简介
数据扩充（data augmentation），又名 数据增强。
其本质即：缺少海量数据时，为了保证模型的有效训练，一分钱掰成两半花。

- 数据扩充方法包括：

简单方法：翻转、旋转、尺度变换、随机抠取、色彩抖动
复杂方法：Fancy PCA、监督式抠取

- 方法介绍

1. 翻转
包括：水平翻转、垂直翻转、水平垂直翻转。OpenCV中的 cv2.flip 接口可用于快速实现翻转操作：opencv: cv2.flip 图像翻转 进行 数据增强

2. 旋转
将原图按照一定角度旋转，作为新图像。
常取的旋转角度为 -30°、-15°、15°、30° 等较刚好的角度值。

3. 尺度变换
将图像分辨率变为原图的0.8、0.9、1.1、1.2等倍数，作为新图像。

4. 抠取
随机抠取：在原图的随机位置抠取图像块，作为新图像。
监督式抠取：只抠取含有明显语义信息的图像块。

5. 色彩抖动
对图像原有的像素值分布进行轻微扰动（即加入轻微噪声），作为新图像。

6. Fancy PCA
对所有训练数据的像素值进行主成分分析（PCA），根据得到的特征向量和特征值计算一组随机值，作为扰动加入到原像素值中。

总之，在实际操作中，常将多种数据扩充操作叠加使用，比如，对原图像分别 (水平、垂直、水平垂直)翻转 和 (-30°、-15°、15°、30°)旋转 后，数据量扩充为原来的8倍。此时，再对这组数据统一各进行一次随机扣取，则数据量翻为原来的16倍。与此类同，我们可以将数据扩充为原来的n次方倍，数据量扩大很多倍
- 注意

- 不是所有 数据扩充方法都可以一股脑儿随便用。比如对于人脸图片，垂直翻转就变得不可行了。因为现实中基本不会出现对倒过来的人脸进行识别，那么垂直翻转后产生的就几乎是对模型有害的噪声了，这会干扰到模型的正常收敛。

- 另外，如果是 图像检测任务 或者是 图像分割任务 ，记得 将 图像数据 和 标记数据 进行 同步扩充（比如图像翻转时，对应的标记坐标跟着做相应翻转），否则扩充后的新图像对应的却是原图像的标记数据。
- 参考文献
[数据扩充 (Data Augmentation)](https://blog.csdn.net/JNingWei/article/details/79219838)

####249.derivative 导数
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-249.jpg)

- 翻译：函数的导数是它变化的速率,直观的来看，某点处的一阶导数是这个点上函数的斜率

导数是函数的局部性质。一个函数在某一点的导数描述了这个函数在这一点附近的变化率。如果函数的自变量和取值都是实数的话，函数在某一点的导数就是该函数所代表的曲线在这一点上的切线斜率。导数的本质是通过极限的概念对函数进行局部的线性逼近。

####255.dropout
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-255.jpg)

- 算法概述
如果要训练一个大型的网络，训练数据很少的话，那么很容易引起过拟合(也就是在测试集上的精度很低)，可能我们会想到用L2正则化、或者减小网络规模。然而深度学习领域大神Hinton，在2012年文献：《Improving neural networks by preventing co-adaptation of feature detectors》提出了，在每次训练的时候，让一半的特征检测器停过工作，这样可以提高网络的泛化能力，Hinton又把它称之为dropout。

Hinton认为过拟合，可以通过阻止某些特征的协同作用来缓解。在每次训练的时候，每个神经元有百分之50的几率被移除，这样可以让一个神经元的出现不应该依赖于另外一个神经元。

另外，我们可以把dropout理解为模型平均。

假设我们要实现一个图片分类任务，我们设计出了100000个网络，这100000个网络，我们可以设计得各不相同，然后我们对这100000个网络进行训练，训练完后我们采用平均的方法，进行预测，这样肯定可以提高网络的泛化能力，或者说可以防止过拟合，因为这100000个网络，它们各不相同，可以提高网络的稳定性。而所谓的dropout我们可以这么理解，这n个网络，它们权值共享，并且具有相同的网络层数(这样可以大大减小计算量)。我们每次dropout后，网络模型都可以看成是整个网络的子网络。(需要注意的是如果采用dropout，训练时间大大延长，但是对测试阶段没影响)。

Dropout说的简单一点就是我们让在前向传导的时候，让某个神经元的激活值以一定的概率p，让其停止工作，示意图如下：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/255_1.png)

左边是原来的神经网络，右边是采用Dropout后的网络。

第一种理解方式是，在每次训练的时候使用dropout，每个神经元有百分之50的概率被移除，这样可以使得一个神经元的训练不依赖于另外一个神经元，同样也就使得特征之间的协同作用被减弱。Hinton认为，过拟合可以通过阻止某些特征的协同作用来缓解。

第二种理解方式是，我们可以把dropout当做一种多模型效果平均的方式。对于减少测试集中的错误，我们可以将多个不同神经网络的预测结果取平均，而因为dropout的随机性，我们每次dropout后，网络模型都可以看成是一个不同结构的神经网络，而此时要训练的参数数目却是不变的，这就解脱了训练多个独立的不同神经网络的时耗问题。在测试输出的时候，将输出权重除以二，从而达到类似平均的效果。

需要注意的是如果采用dropout，训练时间大大延长，但是对测试阶段没影响。

- dropout带来的模型的变化

- 训练层面
无可避免的，训练网络的每个单元要添加一道概率流程。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/255_2.png)
    对应的公式变化如下：

    - 没有dropout的神经网络
    ![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/255_3.png)
    - 有dropout的神经网络

    ![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/255_4.png)

    上面公式中Bernoulli函数，是为了以概率p，随机生成一个0、1的向量。

- 测试层面
预测的时候，每一个单元的参数要预乘以p。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/255_5.png)

- 源码实现

```
#dropout函数的实现
def dropout(x, level):
if level < 0. or level >= 1: #level是概率值，必须在0~1之间
    raise Exception('Dropout level must be in interval [0, 1[.')
retain_prob = 1. - level
#我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
#硬币 正面的概率为p，n表示每个神经元试验的次数
#因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
sample=np.random.binomial(n=1,p=retain_prob,size=x.shape)#即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
print sample
x *=sample # 0、1与x相乘，我们就可以屏蔽某些神经元，让它们的值变为0
print x
x /= retain_prob

return x
#对dropout的测试，可以跑一下上面的函数，了解一个输入x向量，经过dropout的结果
x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
dropout(x,0.4)
#函数中，x是本层网络的激活值。Level就是dropout就是每个神经元要被丢弃的概率
```
- 参考文献：
[理解dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)

#####261.elastic net 弹性网
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-261.jpg)

ElasticNet 是一种使用L1和L2先验作为正则化矩阵的线性回归模型.这种组合用于只有很少的权重非零的稀疏模型，比如:class:Lasso, 但是又能保持:class:Ridge 的正则化属性.我们可以使用 l1_ratio 参数来调节L1和L2的凸组合(一类特殊的线性组合)。

当多个特征和另一个特征相关的时候弹性网络非常有用。Lasso 倾向于随机选择其中一个，而弹性网络更倾向于选择两个。

在实践中，Lasso 和 Ridge 之间权衡的一个优势是它允许在循环过程（Under rotate）中继承 Ridge 的稳定性.
弹性网络的目标函数是最小化:
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/261_1.jpg)
ElasticNetCV 可以通过交叉验证来用来设置参数:
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/261_2.jpg)
- 代码部分如下：

```
import numpy as np
from sklearn import linear_model
import warnings

warnings.filterwarnings('ignore')

###############################################################################
# Generate sample data
n_samples_train, n_samples_test, n_features = 75, 150, 500
np.random.seed(0)
coef = np.random.randn(n_features)
coef[50:] = 0.0  # only the top 10 features are impacting the model
X = np.random.randn(n_samples_train + n_samples_test, n_features)
y = np.dot(X, coef)

# Split train and test data
X_train, X_test = X[:n_samples_train], X[n_samples_train:]
y_train, y_test = y[:n_samples_train], y[n_samples_train:]

###############################################################################
# Compute train and test errors
alphas = np.logspace(-5, 1, 60)
enet = linear_model.ElasticNet(l1_ratio=0.7)
train_errors = list()
test_errors = list()
for alpha in alphas:
    enet.set_params(alpha=alpha)
    enet.fit(X_train, y_train)
    train_errors.append(enet.score(X_train, y_train))
    test_errors.append(enet.score(X_test, y_test))

i_alpha_optim = np.argmax(test_errors)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)

# Estimate the coef_ on full data with optimal regularization parameter
enet.set_params(alpha=alpha_optim)
coef_ = enet.fit(X, y).coef_

###############################################################################
# Plot results functions

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')

# Show estimated coef_ vs true coef
plt.subplot(2, 1, 2)
plt.plot(coef, label='True coef')
plt.plot(coef_, label='Estimated coef')
plt.legend()
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
plt.show()
```
结果如下图所示：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/261_3.png)

- 控制台结果如下：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/261_4.png)

elastic net的大部分函数也会与之前的大体相似，所以这里仅仅介绍一些比较经常用的到的或者特殊的参数或函数：

- 参数：
- **l1_ratio:**在0到1之间，代表在l1惩罚和l2惩罚之间，如果l1_ratio=1，则为lasso，是调节模型性能的一个重要指标。
- **eps:**Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3
- **n_alphas:**正则项alpha的个数
- **alphas：**alpha值的列表

- 返回值：
- alphas：返回模型中的alphas值。
- coefs：返回模型系数。shape=（n_feature,n_alphas）

- 函数：
score（X,y,sample_weight）: 评价模型性能的标准，值越接近1，模型效果越好。

- 参考文献
[弹性网络（Elastic Net）](https://blog.csdn.net/m0_37167788/article/details/78657523)

####267.explained sum of squares 回归平方和
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-267.jpg)

翻译：ess测量的方差(信息)模型

回归平方和，是反映自变量与因变量之间的相关程度的偏差平方和。用回归方程或回归线来描述变量之间的统计关系时，实验值yi与按回归线预测的值Yi并不一定完全一致。

是用于描述一个模型（通常是一个回归模型）如何很好地代表正在建模的数据。具体来说，解释的平方和测量模型值中存在多少变化，并将其与总平方进行比较，所述总平方测量观察数据中存在多少变化，并且与剩余平方和，它测量模型误差的变化。
一般来说，ESS越大，估计模型的性能就越好。
- 参考文献
[explained sum of squares](https://en.wikipedia.org/wiki/Explained_sum_of_squares)

#####273. FPR false positive rate 误检率
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-273.jpg)

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/273_1.png)

首先要搞明白tp,fp,fn,tn分别是什么意思，actual class指的是实际正确的分类，predicted class指的是我们判断的分类。
$t:ture;$
$f:false;$
$p:positive;$
$n:negative$
$p,n是实际结果$
$t,f表示预测结果的真假$

$FPT=\frac{sum(fp)}{sum(fp)+sum(tn)}$

查准率：
$precision-rate=\frac{sum(tp)}{sum(tp)+sum(fp)}$
查全率：
$recall-rate=\frac{sum(tp)}{sum(tp)+sum(fn)}$

误检率是相对于虚假目标的总量里有多少被误识为真实目标；
查准率是指检测到的目标里，真实目标所占的比例；
查全率就是检测到的真实目标，在所有真实目标的比例。
- 参考文献
[false positive rate](https://blog.csdn.net/weixin_41284198/article/details/80391299)

####279.architecture of a neural network 神经网络的结构
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-279.jpg)

翻译：

1. 神经网络的体系结构是指单位,其激活功能,有多少层
2. 大多数的神经网络结构可以理解为单位层的堆积。
3. 解决问题的最佳体系结构应该通过使用验证集进行实验来找到

以监督学习为例，假设我们有训练样本集  $\textstyle (x(^ i),y(^ i)) $，那么神经网络算法能够提供一种复杂且非线性的假设模型 $\textstyle h_{W,b}(x) $，它具有参数 $\textstyle W, b $，可以以此参数来拟合我们的数据。

为了描述神经网络，我们先从最简单的神经网络讲起，这个神经网络仅由一个“神经元”构成，以下即是这个“神经元”的图示：
SingleNeuron.png
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_1.png)
这个“神经元”是一个以 $\textstyle x_1, x_2, x_3 $及截距 $\textstyle +1 $为输入值的运算单元，其输出为 $\textstyle  h_{W,b}(x) = f(W^Tx) = f(\sum_{i=1}^3 W_{i}x_i +b) $，其中函数 $\textstyle f : \Re \mapsto \Re $被称为“激活函数”。我们选用sigmoid函数作为激活函数 $\textstyle f(\cdot)$


$f(z) = \frac{1}{1+\exp(-z)}$.
可以看出，这个单一“神经元”的输入－输出映射关系其实就是一个逻辑回归（logistic regression）。

虽然本系列教程采用sigmoid函数，但你也可以选择双曲正切函数（tanh）：


$f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$,
以下分别是sigmoid及tanh的函数图像
Sigmoid activation function. Tanh activation function.
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_7.png)

$\textstyle \tanh(z) $函数是sigmoid函数的一种变体，它的取值范围为$ \textstyle [-1,1]$ ，而不是sigmoid函数的 $\textstyle [0,1]$ 。

注意，与其它地方（包括OpenClassroom公开课以及斯坦福大学CS229课程）不同的是，这里我们不再令 $\textstyle x_0=1 $。取而代之，我们用单独的参数 $\textstyle b $来表示截距。

最后要说明的是，有一个等式我们以后会经常用到：如果选择 $\textstyle f(z) = 1/(1+\exp(-z))$ ，也就是sigmoid函数，那么它的导数就是 $\textstyle f'(z) = f(z) (1-f(z)) $（如果选择tanh函数，那它的导数就是 $\textstyle f'(z) = 1- (f(z))^2$ ，你可以根据sigmoid（或tanh）函数的定义自行推导这个等式。

- 神经网络模型

所谓神经网络就是将许多个单一“神经元”联结在一起，这样，一个“神经元”的输出就可以是另一个“神经元”的输入。例如，下图就是一个简单的神经网络：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_5.jpg)

我们使用圆圈来表示神经网络的输入，标上“$\textstyle +1$”的圆圈被称为偏置节点，也就是截距项。神经网络最左边的一层叫做输入层，最右的一层叫做输出层（本例中，输出层只有一个节点）。中间所有节点组成的一层叫做隐藏层，因为我们不能在训练样本集中观测到它们的值。同时可以看到，以上神经网络的例子中有3个输入单元（偏置单元不计在内），3个隐藏单元及一个输出单元。

我们用 $\textstyle {n}_l $来表示网络的层数，本例中 $\textstyle n_l=3 $，我们将第 $\textstyle l $层记为 $\textstyle L_l $，于是 $\textstyle L_1 $是输入层，输出层是 $\textstyle L_{n_l}$ 。本例神经网络有参数 $\textstyle (W,b) = (W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}) $，其中 $\textstyle W^{(l)}_{ij}$ （下面的式子中用到）是第 $\textstyle l $层第 $\textstyle j $单元与第 $\textstyle l+1 $层第 $\textstyle i $单元之间的联接参数（其实就是连接线上的权重，注意标号顺序）， $\textstyle b^{(l)}_i$ 是第 $\textstyle l+1 $层第 $\textstyle i$ 单元的偏置项。因此在本例中，$ \textstyle W^{(1)} \in \Re^{3\times 3} $， $\textstyle W^{(2)} \in \Re^{1\times 3} $。注意，没有其他单元连向偏置单元(即偏置单元没有输入)，因为它们总是输出 $\textstyle +1$。同时，我们用 $\textstyle s_l $表示第 $\textstyle l$ 层的节点数（偏置单元不计在内）。

我们用 $\textstyle a^{(l)}_i$ 表示第 $\textstyle l$ 层第 $\textstyle i$ 单元的激活值（输出值）。当 $\textstyle l=1$ 时，$ \textstyle a^{(1)}_i = x_i $，也就是第$ \textstyle i $个输入值（输入值的第 $\textstyle i$ 个特征）。对于给定参数集合 $\textstyle W,b$ ，我们的神经网络就可以按照函数 $\textstyle h_{W,b}(x)$ 来计算输出结果。本例神经网络的计算步骤如下：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_3.jpg)
我们用$ \textstyle z^{(l)}_i $表示第 $\textstyle l $层第 $\textstyle i $单元输入加权和（包括偏置单元），比如， $\textstyle  z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i $，则 $\textstyle a^{(l)}_i = f(z^{(l)}_i) $。

这样我们就可以得到一种更简洁的表示法。这里我们将激活函数 $\textstyle f(\cdot)$ 扩展为用向量（分量的形式）来表示，即 $\textstyle f([z_1, z_2, z_3]) = [f(z_1), f(z_2), f(z_3)] $，那么，上面的等式可以更简洁地表示为：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_4.jpg)
我们将上面的计算步骤叫作前向传播。回想一下，之前我们用 $\textstyle a^{(1)} = x $表示输入层的激活值，那么给定第 $\textstyle l $层的激活值 $\textstyle a^{(l)}$ 后，第 $\textstyle l+1 $层的激活值 $\textstyle a^{(l+1)}$ 就可以按照下面步骤计算得到：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_5.jpg)
将参数矩阵化，使用矩阵－向量运算方式，我们就可以利用线性代数的优势对神经网络进行快速求解。

目前为止，我们讨论了一种神经网络，我们也可以构建另一种结构的神经网络（这里结构指的是神经元之间的联接模式），也就是包含多个隐藏层的神经网络。最常见的一个例子是 $\textstyle  n_l$ 层的神经网络，第 $\textstyle  1$ 层是输入层，第 $\textstyle  n_l $层是输出层，中间的每个层 $\textstyle  l $与层 $\textstyle  l+1 $紧密相联。这种模式下，要计算神经网络的输出结果，我们可以按照之前描述的等式，按部就班，进行前向传播，逐一计算第 $\textstyle  L_2$ 层的所有激活值，然后是第 $\textstyle  L_3 $层的激活值，以此类推，直到第 $\textstyle  L_{n_l}$ 层。这是一个前馈神经网络的例子，因为这种联接图没有闭环或回路。

神经网络也可以有多个输出单元。比如，下面的神经网络有两层隐藏层： $\textstyle L_2 $及 $\textstyle L_3$ ，输出层 $\textstyle L_4 $有两个输出单元。

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/279_6.png)
要求解这样的神经网络，需要样本集 $ \textstyle (x^{(i)}, y^{(i)}) $，其中 $\textstyle y^{(i)} \in \Re^2 $。如果你想预测的输出是多个的，那这种神经网络很适用。（比如，在医疗诊断应用中，患者的体征指标就可以作为向量的输入值，而不同的输出值$ \textstyle y_i$ 可以表示不同的疾病存在与否。）
- 参考文献
[神经网络](http://ufldl.stanford.edu/wiki/index.php/神经网络)



#####285. 基尼指数
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-285.jpg)

20世纪初意大利经济学家基尼，于1922年提出的定量测定收入分配差异程度的指标。它是根据洛伦茨曲线找出了判断分配平等程度的指标（如下图）。

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/285_1.jpg)

　设实际收入分配曲线和收入分配绝对平等曲线之间的面积为A，实际收入分配曲线右下方的面积为B。并以A除以A+B的商表示不平等程度。这个数值被称为基尼系数或称洛伦茨系数。如果A为零，基尼系数为零，表示收入分配完全平等；如果B为零则系数为1，收入分配绝对不平等。该系数可在零和1之间取任何值。收入分配越是趋向平等，洛伦茨曲线的弧度越小，基尼系数也越小，反之，收入分配越是趋向不平等，洛伦茨曲线的弧度越大，那么基尼系数也越大。如果个人所得税能使收入均等化，那么，基尼系数即会变小。

基尼系数的计算公式为：
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/285_2.png)

其中，X代表各组的人口比重，Y代表各组的收入比重，V代表各组累计的收入比重，i=1，2，3，…，n，n代表分组的组数。

- 基尼指数（ CART算法 ---分类树）
定义：基尼指数（基尼不纯度）：表示在样本集合中一个随机选中的样本被分错的概率。

注意： Gini指数越小表示集合中被选中的样本被分错的概率越小，也就是说集合的纯度越高，反之，集合越不纯。

即 基尼指数（基尼不纯度）= 样本被选中的概率 * 样本被分错的概率

书中公式：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/285_3.png)

说明:

1. pk表示选中的样本属于k类别的概率，则这个样本被分错的概率是(1-pk)

2. 样本集合中有K个类别，一个随机选中的样本可以属于这k个类别中的任意一个，因而对类别就加和

3. 当为二分类是，Gini(P) = 2p(1-p)

样本集合D的Gini指数 ： 假设集合中有K个类别，则：

![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/285_4.png)

基于特征A划分样本集合D之后的基尼指数：

需要说明的是CART是个二叉树，也就是当使用某个特征划分样本集合只有两个集合：1. 等于给定的特征值 的样本集合D1 ， 2 不等于给定的特征值 的样本集合D2

实际上是对拥有多个取值的特征的二值处理。

举个例子：

假设现在有特征 “学历”，此特征有三个特征取值： “本科”，“硕士”， “博士”，

当使用“学历”这个特征对样本集合D进行划分时，划分值分别有三个，因而有三种划分的可能集合，划分后的子集如下：

划分点： “本科”，划分后的子集合 ： {本科}，{硕士，博士}
划分点： “硕士”，划分后的子集合 ： {硕士}，{本科，博士}
    划分点： “硕士”，划分后的子集合 ： {博士}，{本科，硕士}
          对于上述的每一种划分，都可以计算出基于 划分特征= 某个特征值 将样本集合D划分为两个子集的纯度：


![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/285_5.png)

因而对于一个具有多个取值（超过2个）的特征，需要计算以每一个取值作为划分点，对样本D划分之后子集的纯度
Gini(D,Ai)，(其中Ai 表示特征A的可能取值)

然后从所有的可能划分的Gini(D,Ai)中找出Gini指数最小的划分，这个划分的划分点，便是使用特征A对样本集合D进行划分的最佳划分点。


- 参考文献
[决策树与基尼指数](https://www.cnblogs.com/muzixi/p/6566803.html)
[决策树之基尼系数](https://blog.csdn.net/qq_16365849/article/details/50644496)

#####291.gradient descent 梯度下降
-by lcx
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/卡片集合/卡片-291.jpg)

- 1. 梯度
在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。比如函数f(x,y), 分别对x,y求偏导数，求得的梯度向量就是(∂f/∂x, ∂f/∂y)T,简称grad f(x,y)或者▽f(x,y)。对于在点(x0,y0)的具体梯度向量就是(∂f/∂x0, ∂f/∂y0)T.或者▽f(x0,y0)，如果是3个参数的向量梯度，就是(∂f/∂x, ∂f/∂y，∂f/∂z)T,以此类推。

那么这个梯度向量求出来有什么意义呢？他的意义从几何意义上讲，就是函数变化增加最快的地方。具体来说，对于函数f(x,y),在点(x0,y0)，沿着梯度向量的方向就是(∂f/∂x0, ∂f/∂y0)T的方向是f(x,y)增加最快的地方。或者说，沿着梯度向量的方向，更加容易找到函数的最大值。反过来说，沿着梯度向量相反的方向，也就是 -(∂f/∂x0, ∂f/∂y0)T的方向，梯度减少最快，也就是更加容易找到函数的最小值。

- 2. 梯度下降法算法详解
- 2.1 梯度下降的直观解释
　　　　首先来看看梯度下降的一个直观的解释。比如我们在一座大山上的某处位置，由于我们不知道怎么下山，于是决定走一步算一步，也就是在每走到一个位置的时候，求解当前位置的梯度，沿着梯度的负方向，也就是当前最陡峭的位置向下走一步，然后继续求解当前位置梯度，向这一步所在位置沿着最陡峭最易下山的位置走一步。这样一步步的走下去，一直走到觉得我们已经到了山脚。当然这样走下去，有可能我们不能走到山脚，而是到了某一个局部的山峰低处。

　　　　从上面的解释可以看出，梯度下降不一定能够找到全局的最优解，有可能是一个局部最优解。当然，如果损失函数是凸函数，梯度下降法得到的解就一定是全局最优解。
![](https://raw.githubusercontent.com/6studentsfromsspku/sspku-300-concepts/master/lcx引用图片/291_1.png)
- 2.2 梯度下降的相关概念
1. 步长（Learning rate）：步长决定了在梯度下降迭代的过程中，每一步沿梯度负方向前进的长度。用上面下山的例子，步长就是在当前这一步所在位置沿着最陡峭最易下山的位置走的那一步的长度。
2. 特征（feature）：指的是样本中输入部分，比如2个单特征的样本$（x^{(0)},y^{(0)}）,（x^{(1)},y^{(1)}）$，则第一个样本特征为$x^{(0)}$，第一个样本输出为$y^{(0)}$。
3. 假设函数（hypothesis function）：在监督学习中，为了拟合输入样本，而使用的假设函数，记为$h_{\theta}(x)$。比如对于单个特征的m个样本$（x^{(i)},y^{(i)}）(i=1,2,...m)$,可以采用拟合函数如下：$h_{\theta}(x) = \theta_0+\theta_1x$
4. 损失函数（loss function）：为了评估模型拟合的好坏，通常用损失函数来度量拟合的程度。损失函数极小化，意味着拟合程度最好，对应的模型参数即为最优参数。在线性回归中，损失函数通常为样本输出和假设函数的差取平方。比如对于m个样本$（x_i,y_i）(i=1,2,...m)$,采用线性回归，损失函数为：$J(\theta_0, \theta_1) = \sum\limits_{i=1}^{m}(h_\theta(x_i) - y_i)^2$

    其中$x_i$表示第i个样本特征，$y_i$表示第i个样本对应的输出，$h_\theta(x_i)$为假设函数。

- 2.3 梯度下降的详细算法
1. 先决条件： 确认优化模型的假设函数和损失函数。
比如对于线性回归，假设函数表示为$h_\theta(x_1, x_2, ...x_n) = \theta_0 + \theta_{1}x_1 + ... + \theta_{n}x_{n}$,(i = 0,1,2... n)为模型参数，$\theta_i $ , $\ x_i $(i = 0,1,2... n)为每个样本的n个特征值。这个表示可以简化，我们增加一个特征$\ x_0 =1$,这样$h_\theta(x_0, x_1, ...x_n) = \sum\limits_{i=0}^{n}\theta_{i}x_{i}$
同样是线性回归，对应于上面的假设函数，损失函数为：$J(\theta_0, \theta_1..., \theta_n) = \frac{1}{2m}\sum\limits_{j=0}^{m}(h_\theta(x_0^{(j)}, x_1^{(j)}, ...x_n^{(j)}) - y_j)^2$

2. 算法相关参数初始化：主要是初始化$\theta_0, \theta_1..., \theta_n$算法终止距离ε
以及步长α。在没有任何先验知识的时候，将所有的θ
初始化为0，将步长初始化为1。在调优的时候再优化。　　　　

3. 算法过程：
1）确定当前位置的损失函数的梯度，对于$\theta_i$,其梯度表达式如下：$\frac{\partial}{\partial\theta_i}J(\theta_0, \theta_1..., \theta_n)$
2）用步长乘以损失函数的梯度，得到当前位置下降的距离，即
$\alpha\frac{\partial}{\partial\theta_i}J(\theta_0, \theta_1..., \theta_n)$对应于前面登山例子中的某一步。
3）确定是否所有的$\theta_i$梯度下降的距离都小于ε
，如果小于ε则算法终止，当前所有的$\theta_i$(i=0,1,...n)即为最终结果。否则进入步骤4.
4）更新所有的θ，对于$\theta_i$，其更新表达式如下。更新完毕后继续转入步骤1.
$\theta_i = \theta_i - \alpha\frac{\partial}{\partial\theta_i}J(\theta_0, \theta_1..., \theta_n)$
下面用线性回归的例子来具体描述梯度下降。假设我们的样本是
$(x_1^{(0)}, x_2^{(0)}, ...x_n^{(0)}, y_0), (x_1^{(1)}, x_2^{(1)}, ...x_n^{(1)},y_1), ... (x_1^{(m)}, x_2^{(m)}, ...x_n^{(m)}, y_m)$,损失函数如前面先决条件所述：$J(\theta_0, \theta_1..., \theta_n) = \frac{1}{2m}\sum\limits_{j=0}^{m}(h_\theta(x_0^{(j)}, x_1^{(j)}, ...x_n^{(j)})- y_j)^2$
则在算法过程步骤1中对于$\theta_i$的偏导数计算如下：
$\frac{\partial}{\partial\theta_i}J(\theta_0, \theta_1..., \theta_n)= \frac{1}{m}\sum\limits_{j=0}^{m}(h_\theta(x_0^{(j)}, x_1^{(j)}, ...x_n^{(j)}) - y_j)x_i^{(j)}$
由于样本中没有x0
上式中令所有的$x_0^{j}$为1.
步骤4中$\theta_i$的更新表达式如下：
$\theta_i = \theta_i - \alpha\frac{1}{m}\sum\limits_{j=0}^{m}(h_\theta(x_0^{(j)}, x_1^{(j)}, ...x_n^{j}) - y_j)x_i^{(j)}$

从这个例子可以看出当前点的梯度方向是由所有的样本决定的，加$\frac{1}{m}$是为了好理解。由于步长也为常数，他们的乘机也为常数，所以这里$\alpha\frac{1}{m}$可以用一个常数表示。

- 参考文献
[梯度下降](https://www.cnblogs.com/pinard/p/5970503.html)
