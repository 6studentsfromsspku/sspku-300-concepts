
目录：
[TOC]
####5. 处理缺失值
edit by zzx
![5](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-5.jpg?raw=true)

- 数据处理中，处理缺失值/异常值是个重要一步。

- 判别常用方法：
    1. 简单统计原理
    对属性值进行一个描述性的统计，从而查看哪些值是不合理的。
    2. 3δ原则
    依正态分布的定义，在默认情况下我们可以认定，距离超过平均值3δ的样本是不存在的。 因此，当样本距离平均值大于3δ，则认定该样本为异常值。
    3. 箱型图分析
    首先定义下上四分位和下四分位。上四分位设为 U，表示的是所有样本中只有1/4的数值大于U ；下四分位设为 L，表示的是所有样本中只有1/4的数值小于L。
    设上四分位与下四分位的差值为IQR，即：IQR=U-L。那么，上界为 U+1.5IQR ，下界为： L - 1.5IQR
    判别方式：大于或小于箱型图设定的上下界的数值即为异常值。

- 一般处理方法：
    1. 删除缺失值。简单，但是不适用于缺失值多的数据集。
    2. 填补缺失值：
      1） 均值法。将数据分成几组，把相应该组均值填补进缺失处。
      2） 随机填补法。
      3） 最近距离填补法。
      4） 回归填补法。回归确立模型，预测填补。
      5） 多重填补法法。基于贝叶斯理论，用EM算法来实现对缺失值进行处理的算法。对每一个缺失值都给M个缺失值，数据集就会变成M个，然后用相同的方法对这M个样本集进行处理，得到M个处理结果，总和这M个结果，最终得到对目标变量的估计。

####11. 如何选择隐藏单元激活函数
edit by zzx
![11](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-11.jpg?raw=true)

- 整流线性单元(ReLu)是隐藏单元极好的默认选择。
因为整流线型单元易于优化，与线性单元十分类似。并且处于激活状态时，导数大且一致。

- 也可选择其他激活函数，组建神经网络训练后，再进行评估。

- 使用ReLu有如下特点:
    1. 对于某些输入，神经元是完全不活跃的。
    2. 对于某些输入，神经元的输出和输入成正比。
    3. 大多数时间，神经元是它们不活跃的状态下进行的操作（即具有稀疏激活）。




####23. 交叉项
edit by zzx
![23](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-23.jpg?raw=true)

- 交互项可以表明一个预测变量对一个相应变量的影响在其他预测变量有不同值的时候，是不同的。它的测试方式是将两个预测变量相乘的项放入模型中。在实际中，如果变量之间有关系的话，那么加入回归项能更好地使模型反映变量之间的关系。使用条件是自变量之间也存在一定联系，但是加入交叉项后是否更好需要进行回归模型的评估才可得知。

####29. 雅可比矩阵
edit by zzx
![29](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-29.jpg?raw=true)

- 向量分析中, 雅可比矩阵是一阶偏导数以一定方式排列成的矩阵, 其行列式称为雅可比行列式。它的重要性在于体现了一个可微方程与给出点的最优线性逼近。 因此, 雅可比矩阵类似于多元函数的导数.

- 假设 f: Rn→Rm 是一个从欧式n维空间转换到欧式m维空间的函数。 这个函数由m个实函数组成: f1(x1,…,xn), …, fm(x1,…,xn). 这些函数的偏导数(如果存在)可以组成一个m行n列的矩阵, 这就是所谓的雅可比矩阵,如上图所示。

- 表示为 Jf(x1,…,xn) , 或者 ∂(f1,…,fm)/∂(x1,…,xn) 。

- 如果p是Rn中的一点, f在p点可微分, 那么在这一点的导数由Jf(p)给出。在此情况下, 由F(p)描述的线性算子即接近点p的F的最优线性逼近, x逼近于p。

    f(x) ≈ f(p) + Jf(p)·(x-p)

####35. Bagging算法，Dropout算法
edit by zzx
![35](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-35.jpg?raw=true)

- Bagging算法

- 是一种集成方法，最基本的思想是通过分别训练几个不同分类器，最后对测试的样本，每个分类器对其进行投票。

- 对于Bagging方法，允许采用相同的分类器，相同的训练算法，相同的目标函数。但是在数据集方面，新数据集与原始数据集的大小是相等的。每个数据集都是通过在原始数据集中随机选择一个样本进行替换而得到的。意味着，每个新数据集中会存在重复的样本。

- 例如下图案例
![35_1](https://img-blog.csdn.net/20170813153102572?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbTBfMzc0NzcxNzU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 想实现一个对数字8进行分类的分类器。此时构造了两个数据集，使用相同的学习算法，第一个分类器学习到的是8的上面那部分而第二个分类器学习的是8的下面那个部分。当我们把两个分类器集合起来的时候，此时的分类才是比较好的。

- Dropout算法

- 可以类比成将许多大的神经网络进行集成的一种Bagging方法。每一个神经网络的训练是非常耗时的，且占用很多内存，训练很多的神经网络进行集合分类就显得太不实际了。 但是，dropout可以训练所有子网络的集合，这些子网络通过去除整个网络中的一些神经元来获得。
![35_2](https://img-blog.csdn.net/20170813154717429?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbTBfMzc0NzcxNzU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 如图所示，加载一个样本到Minibatch，然后随机的采样一个不同的二进制掩膜作用在所有的输出，输入，隐藏节点上。每个节点的掩膜都是独立采样的。采样一个掩膜值为1的概率是固定的超参数。

- 对比：

1. 在Bagging中，所有的分类器都是独立的;而在Dropout中，所有的模型都是共享参数的。
2. 在Bagging中，所有的分类器都是在特定的数据集下训练至收敛;在Dropout中没有明确的模型训练过程。网络都是在一步中训练一次（输入一个样本，随机训练一个子网络）。

####41. Lasso算法变量选择
edit by zzx
![41](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-41.jpg?raw=true)

- Lasso算法则是一种能够实现指标集合精简的估计方法。 通过构造一个罚函数得到一个较为精炼的模型，使得它压缩一些系数，同时设定一些系数为零。因此保留了子集收缩的优点，是一种处理具有复共线性数据的有偏估计。 基本思想是在回归系数的绝对值之和小于一个常数的约束条件下，使残差平方和最小化，从而能够产生某些严格等于0 的回归系数，得到可以解释的模型。

![41_1](https://www.ichdata.com/wp-content/upload/qq%E6%88%AA%E5%9B%BE20161123100451_0_o.jpg)


- 计算方法：
Lasso 的目标函数是凸的，不可导的，传统基于导数（梯度）的方法不可用。

- Lasso 的优点：
    1. 当模型为Sparse的时候，估计准确度高 ；
    2. λ增大时，不重要的变量回归系数^β = 0 ；
    3. Lars的收敛速度为O(n·p^2), 等于OLS 的收敛速度。

- Lasso 不适用于：
    1. 模型不是Sparse的时候；
    2. 变量间高度线性相关的时候。

####47. 线性激活函数
edit by zzx
![47](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-47.jpg?raw=true)

- 该激活函数为 y = x ,输出即输入。为的是在神经网络传递中不改变数字，为最简单基本的激活函数。实际应用如在自适应线性网络中。
![47_1](https://img-blog.csdn.net/20170814190703093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemp1UGVjbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

####53. 逻辑回归&线性回归
edit by zzx
![53](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-53.jpg?raw=true)

- 线性回归

    利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。求解最优解的方法有最小二乘法和梯度下降法。

    优点：结果易于理解，计算上不复杂。
    缺点：对非线性数据拟合不好。
    适用数据类型：数值型和标称型数据。

- 逻辑回归

    逻辑回归是分类当中极为常用的手段。假设有一个二分类问题，输出为y∈{0,1}，而线性回归模型产生的预测值为z=wTx+b是实数值，我们希望有一个理想的阶跃函数来帮我们实现z值到0/1值的转化。Sigmoid Function用一个单调可微的函数实现，如下：
    ![53_1](https://img-blog.csdn.net/20170814193639343?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemp1UGVjbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    之后，把Sigmoid Fuction计算得到的值大于等于0.5的归为类别1，小于0.5的归为类别0。

    网络结构如下：
    ![47_3](https://img-blog.csdn.net/20170814190854846?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemp1UGVjbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 优点：
    1. 实现简单；
    2. 分类时计算量非常小，速度很快，存储资源低；
- 缺点：
    1. 容易欠拟合，一般准确度不太高
    2. 只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；
- 适用数据类型：数值型和标称型数据。

- sklearn包中的逻辑回归算法代码：


```
#Import Library
    from sklearn.linear_model import LogisticRegression
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create logistic regression object
    model = LogisticRegression()
    # Train the model using the training sets and check score
    model.fit(X, y)
    model.score(X, y)
    #Equation coefficient and Intercept
    print('Coefficient: \n', model.coef_)
    print('Intercept: \n', model.intercept_)
    #Predict Output
    predicted= model.predict(x_test)


```

####59. 逆矩阵
edit by zzx
![59](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-59.jpg?raw=true)

- 设A是数域上的一个n阶方阵，若在相同数域上存在另一个n阶矩阵B，使得： AB=BA=I（I为n阶单位阵）。 则称B是A的逆矩阵，记为A-1，而A则被称为可逆矩阵。

- 性质：
    1. 可逆矩阵一定是方阵。
    2. （唯一性）如果矩阵A是可逆的，其逆矩阵是唯一的。
    3. A的逆矩阵的逆矩阵还是A。记作（A-1）-1=A。
    4. 可逆矩阵A的转置矩阵AT也可逆，并且（AT）-1=（A-1）T (转置的逆等于逆的转置）
    5. 若矩阵A可逆，则矩阵A满足消去律。即AB=O（或BA=O），则B=O，AB=AC（或BA=CA），则B=C。
    6. 两个可逆矩阵的乘积依然可逆。
    7. 矩阵可逆当且仅当它是满秩矩阵。

####77. Momentum
edit by zzx
![77](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-77.jpg?raw=true)

- 训练网络时，通常先对网络的初始权值按照某种分布进行初始化。初始化权值操作对最终网络的性能影响比较大，合适的网络初始权值能够使得损失函数在训练过程中的收敛速度更快，从而获得更好的优化结果。但是按照某类分布随机初始化网络权值时，存在一些不确定因素，并不能保证每一次初始化操作都能使得网络的初始权值处在一个合适的状态。不恰当的初始权值可能使得网络的损失函数在训练过程中陷入局部最小值，达不到全局最优的状态。

- momentum 动量能够在一定程度上解决这个问题。momentum 动量是依据物理学的势能与动能之间能量转换原理提出来的。当 momentum 动量越大时，其转换为势能的能量也就越大，就越有可能摆脱局部凹域的束缚，进入全局凹域。momentum 动量主要用在权重更新的时候。

- 一般，神经网络在更新权值时，采用如下公式:

    w = w - learning_rate * dw
引入momentum后，采用如下公式：

    v = mu * v - learning_rate * dw
    w = w + v
其中，v初始化为0，mu是设定的一个超变量，最常见的设定值是0.9。可以这样理解上式：如果上次的momentum(v)与这次的负梯度方向是相同的，那这次下降的幅度就会加大，从而加速收敛。

- Nesterov Momentum
这是对传统momentum方法的一项改进,如下：
![77_1](https://img-blog.csdn.net/20150906103038485)
首先，按照原来的更新方向更新一步（棕色线），然后在该位置计算梯度值（红色线），然后用这个梯度值修正最终的更新方向（绿色线）。上图中描述了两步的更新示意图，其中蓝色线是标准momentum更新路径。

####83. 神经元
edit by zzx
![83](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-83.jpg?raw=true)

- 神经元模型是神经网络中最基本的组成成分。通过对n个输入信号，通过带权重的连接（connection）进行传递，将总的输入与阈值进行比较，通过“激活函数”处理产生输出。

- 自适应线性回归神经元示意图
![83_1](https://img-blog.csdn.net/20170814190703093?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemp1UGVjbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 逻辑回归神经元示意图
![83_2](https://img-blog.csdn.net/20170814190854846?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemp1UGVjbw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


####89. 范数
edit by zzx
![89](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-89.jpg?raw=true)

- 距离的定义是一个宽泛的概念，只要满足非负、自反、三角不等式就可以称之为距离。范数是一种强化了的距离概念，它在定义上比距离多了一条数乘的运算法则。在数学上，范数包括向量范数和矩阵范数，向量范数表征向量空间中向量的大小，矩阵范数表征矩阵引起变化的大小。

- 常见范数：

- L-P范数
L-P范数是一组范数，定义如下：
![89_1](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_1.PNG?raw=true)
根据P 的变化，范数也有着不同的变化，一个经典的有关P范数的变化图如下：
![89_2](https://img-blog.csdn.net/20160623222921977)
实际上，在0≤p<1时，Lp并不满足三角不等式的性质，也就不是严格意义下的范数。

- L0范数
当P=0时，也就是L0范数，由上面可知，L0范数并不是一个真正的范数，它主要被用来度量向量中非零元素的个数。
通常情况下，都是用如下公式：
![89_3](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_3.PNG?raw=true)
其优化问题为
![89_4](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_4.PNG?raw=true)
在实际应用中，由于L0范数本身不容易有一个好的数学表示形式，所以在实际情况中，L0的最优问题会被放宽到L1或L2下的最优化。

- L1范数
定义如下：
![89_5](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_5.PNG?raw=true)
使用L1范数可以度量两个向量间的差异，如绝对误差和:
![89_6](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_6.PNG?raw=true)
其优化问题如下：
![89_7](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_7.PNG?raw=true)

- L2范数
定义如下：
![89_8](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_8.PNG?raw=true)
L2可以度量两个向量间的差异，如平方差和：
![89_9](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_9.PNG?raw=true)
其优化问题如下：
![89_10](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_10.PNG?raw=true)

- L-∞范数
通常情况下，都是用如下公式：
![89_11](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/zzx%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/89_11.PNG?raw=true)

####101. 偏差与方差的权衡
edit by zzx
![101](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-101.jpg?raw=true)

- 偏差:度量了学习算法的期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力。
- 方差：度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响。

- 一般来说，偏差与方差是有冲突的，这称为偏差-方差窘境。如下图所示，给定学习任务，假定我们能够控制学习算法的训练程度，则在训练不足时，学习器的拟合能力不够强，训练数据的扰动不足以使学习器产生显著变化，此时偏差主导了泛化错误率；随着训练程度的加深，学习器的拟合能力逐渐增强，训练数据发生的扰动渐渐能被学习器学到，方差逐渐主导了泛化错误率；在训练程度充分后，学习器的拟合能力已经非常强，训练数据发生的轻微扰动都能导致学习器发生显著变化，若训练数据自身的，非全局的特性被学习器学到了，则发生了过拟合。
![101_1](https://images2015.cnblogs.com/blog/995611/201704/995611-20170401092316461-2061277757.png)

- 测试集的MSE（mean squared error）：
MSE(x) = var(x) + (bias(x))^2 + ϵ^2

- 图形化定义：
![101_2](https://img-blog.csdn.net/20160901113415962)

- 为了达到一个合理的 bias-variance 的平衡，此时需要对模型进行认真地评估。

- 介绍一个有用的cross-validation技术K-fold Cross Validation (K折交叉验证)。

  - K折交叉验证，初始采样分割成K个子样本，一个单独的子样本被保留作为验证模型的数据，其他K-1个样本用来训练。交叉验证重复K次，每个子样本验证一次，、我们便可获得 k 个模型及其性能评价。平均K次的结果或者使用其它结合方式，最终得到一个单一估测。

    当K值大的时候， 我们会有更少的Bias(偏差), 更多的Variance。
    当K值小的时候， 我们会有更多的Bias(偏差), 更少的Variance。

    如下图所示：
    ![101_3](https://img-blog.csdn.net/20160901171029859)

####107. 利用K-NN处理缺失值
edit by zzx
![107](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-107.jpg?raw=true)

- K-NN算法在之前的卡片《K-Nearest Neighbor》做过。

- 使用K-NN算法填补缺失值，即使用与值丢失的属性最相似的属性来估计属性的缺失值，通过距离函数确定两个属性的相似度。

- 优点
    1. K-NN可以预测定性和定量属性
    2. 不需要为缺少数据的每个属性创建预测模型
    3. 具有多个缺失值的属性可以轻松处理
    4. 数据的相关结构被考虑在内

- 缺点
    1. K-NN算法在分析大数据方面非常耗时, 搜索所有数据集，寻找最相似的实例。
    2. k值的选择是非常关键的。 k较高脱显不了显著性属性，而较低的k会丢失重要属性。

- 例如R语言中用K-NN填补缺失值的例子

    require(DMwR)
    knnOutput <- knnImputation(data.copy[c(-1,-4,-5)])
    anyNA(knnOutput)


####125. 数据预处理-训练集和测试集数据划分
edit by zzx
![125](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-125.jpg?raw=true)

- 首先对要放入训练集的数据进行预处理。预处理一般包括如下步骤：
    1. 去除唯一属性。简单删除即可。
    2. 处理缺失值。相应卡片已罗列。
    3. 特征编码。一般为特征二元化、独热编码（采用N位状态寄存器来对N个可能的取值进行编码，每个状态都由独立的寄存器来表示，并且在任意时刻只有其中一位有效）。
    4. 数据标准化、正则化。标准化如min-max标准化、z-score标准化。正则化是将样本的某个范数（如L1范数）缩放到到位1，正则化的过程是针对单个样本的，对于每个样本将样本缩放到单位范数。
    5. 特征选择。进行特征选择的两个主要原因是：减轻维数灾难问题；降低学习任务的难度。常见的特征选择方法分为三类：过滤式（filter）、包裹式（wrapper）、嵌入式（embedding）。
    6. 稀疏表示和字典学习。

- 进行训练集和测试集数据划分，可以使用函为sklearn.cross_validation.train_test_split，例如：


```
import numpy as np
    from sklearn.cross_validation import train_test_split
    X, y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


```



####137. 受试者工作特性曲线
edit by zzx
![137](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-137.jpg?raw=true)

- ROC曲线多用于二类分类器的评估。一个 ROC 曲线，展示了一个分类模型在所有分类阈值下的表现。这张图描绘了两个参数：
 -  True Positive Rate
 -  False Positive Rate

- 定义

TPR=TP/(TP+FN)
FPR=FP/(FP+TN)

- 一条受试者工作特性曲线描绘了真阳率和假阳率在不同分类阈值下的表现。降低分类阈值会把更多的样本标注为阳性，因此真阳率和假阳率同时增加。

- ROC曲线都在y=x的曲线之上，因为y=x表示了随即猜测，不存在比随机猜测更糟糕的机器学习算法，因为总是可以将错误率转换为正确率，如一个算法的正确率为40%，那么将两类的标签互换，正确率就变为了60%。最理想的分类器是到达(0,1)点的折线，现实中不存在。如果说一个机器学习算法A比B好，指的是A的ROC曲线完全覆盖了B的ROC曲线。如果有交点，只能说明A在某个场合优于B。

- 为了计算 ROC 曲线中的点，我们需要在不同的分类阈值下计算逻辑回归模型，但是这效率低。幸运的是，有一个基于排序的高效的算法提供我们这些信息，它叫做AUC。

- AUC

- Area under the ROC Curve，ROC曲线下面积，意味着AUC测量整个ROC曲线下从（0，0）到（1，1）二维区域的面积。

- 线下面积曲线对所有可能的阈值下的表现进行了集中测量。一种AUC的解释是模型把随机阳性样本排在随机阴性样本前面的概率。例如，给定下面的样本，从左往右逻辑回归模型预测值依次递增。AUC代表随机一个正（绿）样本排在随机一个负（红）样本右边的可能性。
![137_1](https://img-blog.csdn.net/20180316103901291)

- AUC的值介于0到1之间。一个模型预测全错时，它的AUC等于0；一个模型预测全对时，它的AUC等于1.0。

####143. 均方根反向传播
edit by zzx
![143](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-143.jpg?raw=true)

- RProp和RMSProp都是一种权值更新算法，类似于SGD算法。

- RProp算法

1. 首先为各权重变化赋一个初始值，设定权重变化加速因子与减速因子。
2. 在网络前馈迭代中当连续误差梯度符号不变时，采用加速策略，加快训练速度；当连续误差梯度符号变化时，采用减速策略，以期稳定收敛。
3. 网络结合当前误差梯度符号与变化步长实现BP，同时，为了避免网络学习发生振荡或下溢，算法要求设定权重变化的上下限。

- 不同权值参数的梯度的数量级可能相差很大，因此很难找到一个全局的学习步长。适用于full-batch learning，不适用于mini-batch learning。

- RMSProp

    RMSProp算法不再孤立地更新学习步长，而是联系之前的每一次梯度变化情况，具体如下。
    1. RMSPprop算法给每一个权值一个变量MeanSquare(w,t)用来记录第t次更新步长时前t次的梯度平方的平均值。
    2. 然后再用第t次的梯度除上前t次的梯度的平方的平均值，得到学习步长的更新比例。
    3. 根据此比例去得到新的学习步长。如果当前得到的梯度为负，那学习步长就会减小一点点；如果当前得到的梯度为正，那学习步长就会增大一点点。

- 这些算法并不能完全解决局部最小值问题，只是使得参数收敛的速度更快。

####149. PCA主成分降维维度选择
edit by zzx
![149](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-149.jpg?raw=true)

- 主成分分析在相应的卡片论述过。

- 设有m条n维数据。

1. 将原始数据按列组成n行m列矩阵X;
2. 将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值;
3. 求出协方差矩阵;
4. 求出协方差矩阵的特征值及对应的特征向量;
5. 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P;
6. 即为降维到k维后的数据。

- PCA的原理是，为了将数据从n维降低到k维，需要找到k个向量，用于投影原始数据，是投影误差（投影距离）最小。用如上图的表达式计算。分子表示原始点与投影点之间的距离之和，而误差越小，说明降维后的数据越能完整表示降维前的数据。如果这个误差小于0.01，说明降维后的数据能保留99%的信息。

- 示例
    用sklearn封装的PCA方法，做PCA的代码如下。


``` from sklearn.decomposition import PCA
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    x=np.array([[10001,2,55], [16020,4,11], [12008,6,33], [13131,8,22]])

    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(x)

    # PCA
    pca = PCA(n_components=0.9)# 保证降维后的数据保持90%的信息
    pca.fit(x)
    pca.transform(x)

```
- 所以在实际使用PCA时，我们不需要选择k，而是直接设置n_components为float数据。

####155. Softmax函数
edit by zzx
![155](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-155.jpg?raw=true)

- 应用场合：
希望分值大的那一项经常取到，分值小的那一项也偶尔可以取到，用softmax可以实现。设a和b，a>b，如果按照softmax来计算取a和b的概率，那a的softmax值大于b的，所以a会经常取到，而b也会偶尔取到，概率跟它们本来的大小有关。

- 定义：
假设有一个数组Z，Zi表示Z中的第i个元素，那么这个元素的Softmax值就是如上图表示。

- 计算与标注样本的差距
在神经网络的计算当中，需要计算按照神经网络的正向传播计算的分数S1，和按照正确标注计算的分数S2之间的差距。计算Loss，才能应用反向传播。Loss定义为交叉熵。
![155_1](https://www.zhihu.com/equation?tex=L_i%3D-log%28%5Cfrac%7Be%5E%7Bf_%7By_i%7D%7D%7D%7B%5Csum_j%7Be%5Ej%7D%7D%29)


- 对分类的Loss进行改进的时候，要通过梯度下降，每次优化一个step大小的梯度。
- 定义选到yi的概率：
![155_2](https://www.zhihu.com/equation?tex=P_%7By_i%7D%3D%5Cfrac%7Be%5E%7Bf_%7By_i%7D%7D%7D%7B%5Csum_j%7Be%5Ej%7D%7D)
- 然后求Loss对每个权重矩阵的偏导，应用链式法则。
![155_3](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7BL_i%7D%7D%7B%5Cpartial%7Bf_%7By_i%7D%7D%7D%3D%5Cfrac%7B%5Cpartial%28-%5Cln%28%5Cfrac%7Be%5E%7Bf_%7By_%7Bi%7D%7D%7D%7D%7B%5Csum_%7Bj%7De%5E%7B%7Bj%7D%7D%7D%29%29%7D%7B%5Cpartial%7Bf_%7By_i%7D%7D%7D%3DP_%7Bf_%7By_i%7D%7D-1)

- 例子：
通过若干层的计算，最后得到的某个训练样本的向量的分数是[ 1, 5, 3 ], 那么概率分别就是[0.015,0.866,0.117]。如果这个样本正确的分类是第二个的话，那么计算出来的偏导就是[0.015,0.866−1,0.117]=[0.015,−0.134,0.117]。

####167. 词干提取
edit by zzx
![167](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-167.jpg?raw=true)

- 在语言形态学和信息检索里，词干提取是去除词缀得到词根的过程─—得到单词最一般的写法。对于一个词的形态词根，词干并不需要完全相同；相关的词映射到同一个词干一般能得到满意的结果，即使该词干不是词的有效根。

- 以下介绍使用python实现的NLTK分词器。

- 创建一个去除标点符号等特殊字符的正则表达式分词器。


```    import nltk tokenizer = nltk.RegexpTokenizer(r'w+')
对准备好的数据表进行处理，添加词干将要写入的列，以及统计列，预设默认值为1。

    df["Stemming Words"] = "" df["Count"] = 1
读取数据表中的Words列，使用波特词干提取器取得词干。

    j = 0 while (j <= 5): for word in tokenizer.tokenize(df["Words"][j]): df["Stemming Words"][j] = df["Stemming Words"][j] + " " + nltk.PorterStemmer().stem_word(word) j += 1 df
进行分组统计。

    uniqueWords = df.groupby(['Stemming Words'], as_index = False).sum().sort(['Count']) uniqueWords
拼写检查，针对Python我们可以使用enchant。

    sudo pip install enchant
    import enchant from nltk.metrics import edit_distance class SpellingReplacer(object): def __init__(self, dict_name='en', max_dist=2): self.spell_dict = enchant.Dict(dict_name) self.max_dist = 2 def replace(self, word): if self.spell_dict.check(word): return word suggestions = self.spell_dict.suggest(word) if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist: return suggestions[0] else: return word from replacers import SpellingReplacer replacer = SpellingReplacer() replacer.replace('insu') 'insu'
对已有的结果进行相似度计算，将满足最小偏差的数据归类到相似集中。

    import Levenshtein minDistance = 0.8 distance = -1 lastWord = "" j = 0 while (j < 1): lastWord = uniqueWords["Stemming Words"][j] distance = Levenshtein.ratio(uniqueWords["Stemming Words"][j], uniqueWords["Stemming Words"][j + 1]) if (distance > minDistance): uniqueWords["Stemming Words"][j] = uniqueWords["Stemming Words"][j + 1] j += 1 uniqueWords
重新对数据结果进行分组统计。

    uniqueWords = uniqueWords.groupby(['Stemming Words'], as_index = False).sum() uniqueWords
至此完成了初步的词干提取处理。
```

####173. 指导深度学习的经验法则
edit by zzx
![173](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-173.jpg?raw=true)

- 为了得到好的结果，每个目标类别中的观测值不少于5000个。
- 总体的观测值不少于10,000,000个。

####179. 容量
edit by zzx
![179](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-179.jpg?raw=true)

- 容量从本质上说是描述了整个模型的拟合能力的大小。

- 如果容量不足，模型将不能够很好地表示数据，表现为欠拟合；如果容量太大，那么模型就很容易过分拟合数据，因为其记住了不适合与测试集的训练集特性，表现为过拟合。因此控制好模型的容量是一个关键问题。

- 虽然更简单的函数更可能泛化（训练误差和测试误差的差距小），但我们仍然需要选择一个充分复杂的假设以达到低的训练误差。通常，当模型容量上升时，训练误差会下降，直到其渐近最小可能误差（假设误差度量有最小值）。通常， 泛化误差是一个关于模型容量的 U 形曲线函数。
![179_1](https://img-blog.csdn.net/20170927085930854?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTG9zZUluVmFpbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 容量的控制一般有两种方法：
 - 控制模型的假设空间。
 - 添加正则项对模型进行偏好排除。

- 假设空间(hypothesis space)指的是算法可以作为解决方案的函数集合，比如在线性回归中，广义线性回归模型对线性回归模型的补充，则就扩大了模型的容量，增加了其表达能力，也使得其更容易过拟合。
![179_2](https://img-blog.csdn.net/20170927085958480?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTG9zZUluVmFpbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

####185. Dropout对隐藏单元的影响
edit by zzx
![185](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-185.jpg?raw=true)

使用Dropout算法，神经网络的训练和预测就会发生一些变化。

- 训练层面
    训练网络的每个单元要添加一道概率流程。
    ![185_1](https://img-blog.csdn.net/20160917202613792)
    公式也相应发生变化
    -  没有Dropout的神经网络
    ![185_2](https://img-blog.csdn.net/20160917202649650)
    - 有Dropout的神经网络
    ![185_3](https://img-blog.csdn.net/20160917202715761)

- 测试层面
    预测的时候，每一个单元的参数要预乘以p。
    ![185_4](https://img-blog.csdn.net/20160917202747355)

####191. 阈值激活函数
edit by zzx
![191](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-191.jpg?raw=true)

- 常用的线性激活函数的一种。
![191_1](https://img-blog.csdn.net/20180406160630892)

- 常见的还包括如下两种：
    - 线性函数 ( Liner Function )
    ![191_2](https://img-blog.csdn.net/20180406160600275)
    - 斜面函数 ( Ramp Function )
    ![191_3](https://img-blog.csdn.net/20180406160613687)

####197. Training Error Rate
edit by zzx
![197](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-197.jpg?raw=true)

定义为预测失败个数占总预测个数的比例，评估训练模型的预测能力。可以以如图中的0-1损失函数进行计算。

####203. 均匀分布
edit by zzx
![203](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-203.jpg?raw=true)

- 一个均匀分布在区间[a,b]上的连续型随机变量X 可给出如下函数。
概率密度函数：
![203_1](https://gss3.bdstatic.com/7Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D152/sign=01c7c7300d7b020808c93be450d8f25f/500fd9f9d72a60594aec0e8f2f34349b023bbaa3.jpg)
- 累计分布函数：
![203_2](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D154/sign=4e6ce61c0ae939015202893b4fed54f9/d009b3de9c82d158fe0a078d8b0a19d8bc3e4205.jpg)

- 均值：
![203_3](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D301/sign=bc27d07159df8db1b82e7a643822dddb/b812c8fcc3cec3fd1c44769cd188d43f869427e5.jpg)
- 方差
![203_4](https://gss0.bdstatic.com/94o3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D111/sign=1762c83490504fc2a65fb404d4dde7f0/a71ea8d3fd1f4134af4641422e1f95cad1c85e9c.jpg)

- 如果X〜U（a，b）并且[x，x + d]是具有固定d> 0的[a，b]的子间隔，则
![203_5](https://gss3.bdstatic.com/7Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D263/sign=dd29113407d79123e4e093729e355917/8d5494eef01f3a293a5c94f29225bc315c607c71.jpg)

####209. 特征选择——方差选择法
edit by zzx
![209](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-209.jpg?raw=true)

- 移除方差比较低的特征，使用方差作为特征选择的一个标准，是因为观察到这么一个事实，方差较低的样本差异不大，对我们的目标变量贡献比较低，所以我们移除方差比较低的样本。

- 示例:
    假如特征是boolean类型，那么它是伯努利随机变量，它的方差为D(X)=p(1−p)。 假如我们想要移除特征当中有超过80%要么是0要么是1的样本，那么我们把方差的阈值定义为 0.8*（1-0.8）=0.16。

```
        from sklearn.feature_selection import VarianceThreshold
        X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
        clf = VarianceThreshold(threshold=(0.8 * (1-0.8)))
        print(clf.fit_transform(X))
```

输出结果，发现将第一列移除了。


####215. 权值衰减
edit by zzx
![215](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-215.jpg?raw=true)

- 在机器学习中，常常会出现过拟合，网络权值越大往往过拟合的程度越高。因此，为了避免出现过拟合，会给误差函数添加一个惩罚项，常用的惩罚项是所有权重的平方乘以一个衰减常量之和。
![215_1](http://ufldl.stanford.edu/wiki/images/math/4/5/3/4539f5f00edca977011089b902670513.png)

- 应用此方法的回归称为岭回归。

- 岭回归(Ridge Regression)是在平方误差的基础上增加正则项：
![215_2](http://latex.codecogs.com/gif.latex?%5Csum_%7Bi=1%7D%5E%7Bn%7D%5Cleft&space;(&space;y_i-%5Csum_%7Bj=0%7D%5E%7Bp%7Dw_jx_%7Bij%7D&space;%5Cright&space;)%5E2+%5Clambda&space;%5Csum_%7Bj=0%7D%5E%7Bp%7Dw%5E2_j)

- 随着的增大，模型方差减小而偏差增大，需调整至平衡。

- 对求导，结果为
![215_3](http://latex.codecogs.com/gif.latex?2X%5ET%5Cleft&space;(&space;Y-XW&space;%5Cright&space;)-2%5Clambda&space;W)
- 令其为0，可求得的值
![avatar](http://latex.codecogs.com/gif.latex?%5Chat%7Bw%7D=%5Cleft&space;(&space;X%5ETX+%5Clambda&space;I&space;%5Cright&space;)%5E%7B-1%7DX%5ETY)

####221. 约登指数
edit by zzx
![221](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-221.jpg?raw=true)

- 定义式
![221_1](https://wikimedia.org/api/rest_v1/media/math/render/svg/48eb51856feb1364cf3d91a44f9ec73b6f0a616d)
- 可表达为
![221_2](https://wikimedia.org/api/rest_v1/media/math/render/svg/db350d81cb3c935786da783e16bf66a0b287d8a2)

- 指数范围[ -1 , 1 ]，数值越大，筛查实验的效果越好，真实性越大。

####227. 组合
edit by zzx
![227](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-227.jpg?raw=true)

- 一般地，从n个不同的元素中，任取k（k≤n）个元素为一组，叫作从n个不同元素中取出k个元素的一个组合。我们把有关求组合的个数的问题叫作组合问题。

- 一些性质：
![227_1](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D83/sign=fb5a180b9ecad1c8d4bbf1247f3e115a/2f738bd4b31c87016a9fbac22c7f9e2f0708ffed.jpg)
![227_2](https://gss1.bdstatic.com/9vo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D192/sign=34edecdcac86c9170c035630fb3d70c6/0eb30f2442a7d93369b2ec57a64bd11373f001ff.jpg)
![227_3](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D199/sign=e095ecb607f431adb8d247307236ac0f/3b292df5e0fe9925ee1360743fa85edf8db171f2.jpg)

####239. 交叉熵
edit by zzx
![239](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-239.jpg?raw=true)

- 定义
![239_1](https://images2017.cnblogs.com/blog/1160281/201710/1160281-20171015181224230-912519156.png)
p为真实的概率分布，q为预测的概率分布。

- 引入 KL散度D(p||q)  = H(p,q) - H(p) = ![239_2](https://images2017.cnblogs.com/blog/1160281/201710/1160281-20171015182210965-1841254544.png),也叫做相对熵，它表示两个分布的差异，差异越大，相对熵越大。

- Cross-Entroy 损失函数
![239_3](https://img-blog.csdn.net/20171016222123719?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMjU1NTI1Mzk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 与方差损失函数相似的性质：

1. 损失函数永远大于0。
2. 计算值与真实值越接近，损失函数越小。两者差距越大，损失函数越大。

- 相比方差损失函数的优点：
    当误差大的时候，权重更新就快，当误差小的时候，权重的更新就慢。

    示例：


    ```
    # y_ 真实输出值，y 预测值
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_ent = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(y), reduce_indices=[1]))

    ```
####245. 决策边界
edit by zzx
![245](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-245.jpg?raw=true)

- 用以对不同类别的数据分割的边界，边界的两旁应该是不同类别的数据。

- 以sigmoid函数为例：
    当g(z)≥0.5时, z≥0，对于hθ(x)=g(θTX)≥0.5, 则θTX≥0, 此时意味着预估y=1;
    反之，当预测y = 0时，θTX<0;
    所以我们认为θTX =0是一个决策边界，当它大于0或小于0时，逻辑回归模型分别预测不同的分类结果。

- 例如hθ(x)=g(θ0+θ1X1+θ2X2)，其中θ0 ,θ1 ,θ2分别取-3, 1, 1。则当−3+X1+X2≥0时, y = 1; 则X1+X2=3是一个决策边界，图形表示如下，刚好把图上的两类点区分开来：
![245_1](https://img-blog.csdn.net/20151014124638710?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 假设我们的数据呈现出如下图的分布情况，函数图像为一个圆，圆点在原点且半径为1，这样一条曲线来分隔开了 y=1 和 y=0 的区域，所以需要的是一个二次方特征。
![245_2](https://img-blog.csdn.net/20160425104808726)
![245_3](https://img-blog.csdn.net/20160425104959633)

####251. 行列式
edit by zzx
![251](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-251.jpg?raw=true)

- 行列式在数学中，是一个函数，其定义域为det的矩阵A，取值为一个标量，写作det(A)或 | A | 。行列式可以看做是有向面积或体积的概念在一般的欧几里得空间中的推广。或者说，在 n 维欧几里得空间中，行列式描述的是一个线性变换对“体积”所造成的影响。

- 设n阶方阵，其行列式
![251_1](https://gss1.bdstatic.com/9vo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D220/sign=ac2279caa451f3dec7b2be66a4eef0ec/6609c93d70cf3bc7ae097cecda00baa1cd112aa0.jpg)
- 可表示为
![251_2](https://gss3.bdstatic.com/7Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D250/sign=2801c6998926cffc6d2ab8b789014a7d/63d0f703918fa0ec97bed40d2d9759ee3d6ddbb1.jpg)

####257. 岭参数
edit by zzx
![257](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-257.jpg?raw=true)

- 岭回归定义式
![257_1](https://img-blog.csdn.net/20171111002700013?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRGFuZ19ib3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
- 在原先的A的最小二乘估计中加一个小扰动λI，原先无法求广义逆的情况变成可以求出其广义逆，使得问题稳定并得以求解。

- 性质
    1. 当岭参数为0，得到最小二乘解。
    2. 当岭参数λ趋向更大时，岭回归系数A估计趋向于0。
    3. 岭回归是回归参数A的有偏估计。它的结果是使得残差平和变大，但是会使系数检验变好。
    4. 在认为岭参数λ是与y无关的常数时，是最小二乘估计的一个线性变换，也是y的线性函数。
    但在实际应用中，由于λ总是要通过数据确定，因此λ也依赖于y、因此从本质上说，并非的线性变换，也非y的线性函数。
    5. 对于回归系数向量来说，有偏估计回归系数向量长度<无偏估计回归系数向量长度。即![257_2](https://img-blog.csdn.net/20171111002747563?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRGFuZ19ib3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
    6. 存在某一个λ，使得它所对应的的MSE（估计向量的均方误差）<最小二乘法对应估计向量的的MSE。即存在λ>0，使得![257_3](https://img-blog.csdn.net/20171111002751495?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRGFuZ19ib3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 岭迹图
    ![257_4](https://img-blog.csdn.net/20171111002756500?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvRGFuZ19ib3k=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

    - 岭迹图的横坐标为λ，纵坐标为A(λ)。而A(λ)是一个向量，由a1(λ)、a2(λ)、...等很多分量组成，每一个分量都是λ的函数，将每一个分量分别用一条线。当不存在奇异性时，岭迹应是稳定地逐渐趋向于0。

    - 在λ很小时，A很大，且不稳定，当λ增大到一定程度时，A系数迅速缩小，趋于稳定。

- 岭参数选择
    1. 各回归系数的岭估计基本稳定；
    2. 用最小二乘估计时符号不合理的回归系数，其岭估计的符号变得合理；
    3. 回归系数没有不合乎实际意义的值；
    4. 残差平方和增大不太多。 一般λ越大，系数β会出现稳定的假象，但是残差平方和也会更大。

####263. 编码有序特征
edit by zzx
![263](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-263.jpg?raw=true)

- 有序分类变量的量化，常用顺序编号代替具体的分类来量化。

####269. 梯度爆炸问题
edit by zzx
![269](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-269.jpg?raw=true)

- 深层网络由许多非线性层堆叠而来，每一层非线性层都可以视为是一个非线性函数 f(x)(非线性来自于非线性激活函数），因此整个深度网络可以视为是一个复合的非线性多元函数。

- 最终的目的是希望这个多元函数可以很好的完成输入到输出之间的映射，假设不同的输入，输出的最优解是g(x) ，那么，优化深度网络就是为了寻找到合适的权值，满足Loss=L(g(x),F(x))取得极小值点。对于这种数学寻找最小值问题，采用梯度下降的方法再适合不过了。

- 前面层上的梯度是来自于后面层上梯度的乘积。当存在过多的层次时，就出现了内在本质上的不稳定场景，如梯度消失和梯度爆炸。

- 例如三个隐层的单神经元网络
![269_1](https://img-blog.csdn.net/20170401100429934?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3BwamF2YV8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
- 可以得到表达式
![269_2](https://img-blog.csdn.net/20170401100440508?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3BwamF2YV8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 同时，sigmoid方程的导数曲线
![269_3](https://img-blog.csdn.net/20170401100448622?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3BwamF2YV8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- sigmoid导数的最大值为1/4，通常abs（w）<1,则
![269_4](https://img-blog.csdn.net/20170401100501836?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3BwamF2YV8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
- 前面的层比后面的层梯度变化更小，故变化更慢，从而引起了梯度消失问题。

- 梯度消失与梯度爆炸其实是一种情况，当权值过大，前面层比后面层梯度变化更快，会引起梯度爆炸问题。

- 解决方案：
    1. 预训练加微调；
    2. 梯度剪切、正则；
    3. 使用Relu、LeakRelu、Elu等激活函数替代。


####275. 特征选择
edit by zzx
![275](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-275.jpg?raw=true)

- 特征选择的方法是从原始特征数据集中选择出子集，是一种包含的关系，没有更改原始的特征空间。

- 目标
    - 提高预测的准确性
    - 构造更快，消耗更低的预测模型
    - 能够对模型有更好的理解和解释

- 方法

    Filter方法
    主要思想：对每一维的特征“打分”，即给每一维的特征赋予权重，这样的权重就代表着该维特征的重要性，然后依据权重排序。
    - Chi-Squared Test(卡方检验)
    - Information Gain(信息增益)
    - Correlation Coefficient Scores(相关系数)

    Wrapper方法
    主要思想：将子集的选择看作是一个搜索寻优问题，生成不同的组合，对组合进行评价，再与其他的组合进行比较。这样就将子集的选择看作是一个是一个优化问题，这里有很多的优化算法可以解决，尤其是一些启发式的优化算法，如GA，PSO，DE，ABC等.

    Embedded方法
    主要思想：在模型既定的情况下学习出对提高模型准确性最好的属性。在确定模型的过程中，挑选出那些对模型的训练有重要意义的属性。
    主要方法：正则化，例如岭回归就是在基本线性回归的过程中加入了正则项。

####281. 弗罗贝尼乌斯范数
edit by zzx
![281](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-281.jpg?raw=true)

- 定义式
![281_1](http://upload.wikimedia.org/math/3/9/a/39a998ba0e24ee854346ed2575cc1de0.png)
- 这里 A* 表示 A 的共轭转置，σi 是 A 的奇异值，并使用了迹函数。弗罗贝尼乌斯范数与 Kn 上欧几里得范数非常类似，来自所有矩阵的空间上一个内积。

- 弗罗贝尼乌斯范范数是服从乘法的且在数值线性代数中非常有用。这个范数通常比诱导范数容易计算。

- 此范数针对矩阵，用于比较矩阵大小。
