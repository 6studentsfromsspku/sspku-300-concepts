####6.黑塞矩阵
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-6.jpg?raw=true)
什么是黑塞矩阵
黑塞矩阵是一个多元函数的二阶偏导构成的方阵。
黑塞矩阵的具体描述
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/6_1.png?raw=true)
输入一个多元函数f，H(f)中的第(i,j)元素表示的是函数f对变量Xi和Xj的二次偏导数。
黑塞矩阵的意义：多元函数对各变量的一阶导数能求得函数特殊点（极值点或是鞍点），然而要具体判断该点是极值点还是鞍点要通过二阶导数来判断，对这个黑塞矩阵求行列式，若|H|>0则为极值点，若|H|<0则为鞍点，若|H|=0则无法判断，还需要更高阶的导数才能进行判断。
####12.调参数
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-12.jpg?raw=true)
调参数
什么是超参数
模型当中，需要自己预先设置的参数
如何调参
调整超参数，以获得最佳模型。
没有统一的调整参数的方法。
如果是小样本小模型的话可以用grid_serch来做。
但对于大样本调参更多靠的是经验以及自身对算法的理解程度。
####18.填充缺失值
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-18.jpg?raw=true)
1.如果是数值型的特征，可以通过取平均值来进行填充。
2.如果是类别型的特征，可以通过取最大类别来进行填充。
3.使用模型来预测这个缺失值，例如使用K-means聚类方法来进行填充。
用K-means来进行填充的具体步骤：
1.对其他非缺失值的特征进行聚类。
2.聚在一块后取该类别中该缺失的特征的平均值或者最大类别进行填充。
对缺失值的处理的方法非常多，基本也没有统一的方法，有些比赛对缺失值的良好处理，能极大影响最后的得分。
####24.BAG of WORDS
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-24.jpg?raw=true)
词袋法。
一个词转换成一个计算机能够理解的向量（数组），最原始的方法就是词袋法。
one-hot，若原文本有20000个不同的词，则每个词表示为一个19999个0一个1的向量，其中的1表示命中的该词，因为一个词在20000个特征中只有一个命中，所以才叫one-hot。
####30.joins
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-30.jpg?raw=true)
对两张表格的合并是pandas一个很重要的操作。
concat（强性合并）
concat既可以进行行合并也可以进行列合并，默认情况下是行合并，如果想要列合并就设置axis=1就可以了。
强性合并的意识就是，concat只管合并，并不会做任何处理
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/30_1.png?raw=true)

```
result=pd.concat([df1,df2,df3])
```

merge（软性合并）
merge是对两个Dataframe中某一个列的相同值进行合并。

```
pd.merge(df1,df2,on='Key')
```

![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/30_2.png?raw=true)
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/30_3.png?raw=true)
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/30_4.png?raw=true)
join(merge的特殊版本)
merge一般是基于某一column的相同值进行合并。
join是基于index值来进行合并。
要注意的是，df1和df2不能有相同的column。
join和merge本质是一样的，都是拿一列来进行软性合并，在merge中，这一列是column，在join中，这一列是index。
最后谈谈how这个参数
首先how这个是指两张表参与合并的key值，最后出现在合并表上的方法。
第一个选项：inner，inner是默认选项，最后表格呈现的是A表和B表都有的key值
第二个选项：outer，outer是指全集，最后表格呈现的是A表出现的所有值加上B表出现的所有值，可以说是全key，outer会出现很多很多缺失值。
第三个选项：left，左边的表全部key。
第四个选项：right，右边的表全部key。
####42.leaky relu
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-42.jpg?raw=true)
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/42_1.png?raw=true)
传统的relu是输入小于0是取0，大于0时取本身，导数要么是0要么是1，这个特性非常的棒，很大程度缓和了神经网络更新权值时梯度梯度消失的问题，这也是relu取得巨大成功的原因。
leaky relu是为了解决relu出现的一些问题而提出的，然而leaky relu用的人非常少，不同于relu导数要么0要么1的特性，leaky relu的导数取值要么是1要么是一个较小（这个数可以自己设定）的数。
relu出现的这个问题是若一个靠后的层含有一个很大的负偏差值b，则前面与其链接的神经元的梯度会一直是0，永远没法更新。leaky relu能解决这个问题，然而又带来了新的问题，就是训练慢。
更具体的解释可以参考如下链接：
https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning
####48.向量的线性变换
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-48.jpg?raw=true)
对一个向量乘于一个常数实现变换。
这个操作在神经网络中大量的出现，很多对神经网络的加速方法都是在尝试加快向量线性变换的速度。
####54.逻辑回归
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-54.jpg?raw=true)
逻辑回归实质分为两步
第一步：线性变换，y=w1x1+w2x2+w3x3+................，第一步线性变化得到的y值值域为负无穷大到正无穷大。
第二步：压缩y值，![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/54_1.png?raw=true)，这个函数也叫sigmoid函数，会将输入进去的y值，压缩到值域为（0,1）
####60.matrix multiplication
by-wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-60.jpg?raw=true)
矩阵相乘
![](1528705518232.png)
####66.minibatch
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-66.jpg?raw=true)
最原始进行训练的时候是一个一个样本来进行前向传播然后计算loss，接着反向传播更新权值，这样子做很稳定，但是计算机本身有并行处理的优势，为了充分利用这个优势，提出了minibatch这个概念，就是多个样本，例如128个样本一块进行前向传播计算一个平均loss，然后反向传播更新权值，128个样本一起进行前向传播可以利用计算机的并行性提高了计算效率。但是问题就是，每一轮次更新权值的次数少了很多，虽然这可以通过增加训练轮次来解决。
minibatch越大训练越不稳定，越小则训练效率相对会低，如果训练样本5000以上，我通常会设置minibatch为128。
####72.Missing completely at random
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-72.jpg?raw=true)
缺失值完全随机
什么是完全随机的缺失值
这个概念是指，数据集中的缺失值完全是随机的。
若认为缺失值是完全随机的话，而且占总体比重不大，则可以考虑直接把这些缺失值扔掉。
固定类别的缺失值
但如果是某一类别的数据缺失值特别多，那么扔掉这些缺失值会带来很严重的问题，就是我们的模型被办法掌握这类数据能够带来的信息，当有一个新样本是这类数据的时候则没法进行准确判断。
对于这种缺失值的处理，要非常的谨慎，通常要加入大量的人工推断，才能填补这些缺失值。
例子
有大量的北京天气情况，数据集包含一天24个小时的天气情况记录，要预测北京的PM2.5值，但是发现每天都有一个时间点缺失，例如每天都没有早上6点的数据，这个时候，就不能随便把缺失值给扔了，因为这个可能是问题的关键。另外，如果是没10天都有一个时间点的温度指标没有数据，而且，多个缺失值之间是完全没有联系的，则可以认为，这是完全随机的缺失值，可以直接扔掉。

####78.Motivation for Deep layers
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-78.jpg?raw=true)
为什么要有深层的神经网络。
神经网络为什么要深？这个是一个极其具有深度的问题，我第一次接触这个问题是在看吴恩达网课里面，吴恩达花了10分钟通过两个例子简单的描述了一下，深层的网络能够带来更好的效果，但是吴恩达并没有深入去解释这个问题，现在回头想想，是因为这个问题本身就是一个很具有深度的问题，后来我在寒假的时候看了一篇日本的博士写的一篇文章https://zhuanlan.zhihu.com/p/22888385，非常的有感触，大概理解了多层带来的直接影响就是更多次的非线性变换，但是其实还是充满疑惑，后来周志华在2018年中的时候又提到了这个问题，http://baijiahao.baidu.com/s?id=1597878216780650852&wfr=spider&for=pc，从特征学习的角度去回答这个问题，也是非常非常的深刻，周老师甚至否定了自己一年前的回答，认为自己以前的认识是不完善的，面对这样一个问题连周志华老师都要花两次才能满意的回答上，足以说明这是一个很有深度的问题。
####84.No free lunch theorem
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-84.jpg?raw=true)
没有免费午餐定理
这个定理指的是想象无数个分布，每一个机器学习算法对所有的分布进行训练并预测，得到的平均损失是完全一致的。
####90.Bayersian methonds
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-90.jpg?raw=true)
贝叶斯方法并不是指某一个具体的算法，而是一整类基于贝叶斯条件概率思想的算法，这些算法都有一些共同的特点，如下：
优点：在小样本的时候表现出色并且非常直观
缺点：在大样本底下会消耗很大的计算量
####96.Notions of probability
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-96.jpg?raw=true)
简单来说，Bayesian 与 Frequency 的区别在于对参数的理解不同，具体来说，Frequency视角下，参数是一个实数或者向量，所以参数空间是欧式空间或其子集；而Bayesian视角下，参数是一个随机变量或者随机向量。实数和随机变量的本质区别在于其上能不能建立起概率结构。
一般来说，经典统计问题都可以从Bayesian 与 Frequency 的角度下进行分析，比如有一般的统计推断，也有对应的Bayesian 推断；有一般的假设检验，也有对应的Bayesian 检验。

参考资料：https://www.zhihu.com/question/55819120/answer/146430280
####102.one-sided label smoothing
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-102.jpg?raw=true)
单标签平滑
数据集本身带有的问题：数据可能自身的标签就是错误的，根据这个训练集进行训练，会得到错误的模型。
解决方法：把分类为1的改变为0.9，把分类为0的改变为0.1，这样子就可以减少模型对标签的依赖程度。
上面这个解决方法就是单标签平滑。


####114.Parameters VS Hyperparateters
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-114.jpg?raw=true)
parameters：是指要学习的参数，例如神经网络中每个神经元带有的w1、w2.....、b就是parameters。
hyperparameters：超参数，这个是自己人为调节，例如学习率、神经网络的层数、每层神经元数就是超参数。
####120.platt scaling to create probabilities for svc
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-120.jpg?raw=true)
platt scaling就是衡量一个模型输出来的结果概率是正确

####132.Random Forest
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-132.jpg?raw=true)
随机森林
什么是随机森林
随机森林是集成学习中bagging最直接的应用。其中基学习器是决策树。
第一步：对整体训练集抽出一部分部分数据作为子集。每一颗决策树通过这一子集进行训练，生成决策树。
第二步：每一颗决策树对测试集的样本就行测试得到结果。
第三步：每一颗树给出的结果进行汇总，最多票数即为最终答案。
为什么随机森林表现比决策树要好
这个是一个非常有意思的点，为什么随机森林几乎都比单决策树要好，到底是问什么，在周志华的机器学习里面，周老师用一句很简单的话就说明了原因。
好而不同
周老师书里面就是说了这4个字，然后在书里对这简单的四个字进行了大量的解释，非常有意思。
1. 好：指每一颗决策树都应该表现出色。
2. 而：指的是并且、and、＆、∩。
3. 不同：这个我认为是最关键的就是，每一棵树都应该有自己擅长的方面，也有自己不擅长的方面，这其实也解释了为什么随机森林的训练每科子树应该选用不同的训练集。
####138.Regression
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-138.jpg?raw=true)
回归
什么是回归什么是分类
有监督的机器学习问题大致分为两类，第一类是分类问题，第二类是回归问题。
分类：最后输出的值是类别。
回归：最后输出的是连续值。
回归和分类的本质
实质上分类和回归的本质是一样的，关于这个问题我思考了很久。
例如逻辑回归就是一个很好的例子，逻辑回归本质上是广义的线性回归。用逻辑回归来进行二分类是因为逻辑回归最后要经过1/(1+exp(-y))，这个过程实质上就是做了一个压缩，将线性回归输出的值域在负无穷到正无穷的值压缩到（0,1），这就让模型从处理回归问题变为处理分类问题。再例如神经网络，理论上神经网络的输出都是值域为负无穷到正无穷的连续值，如果要处理二分类问题则最后的输出连接一个激活函数为sigmoid的神经元就可以了，如果是要处理多分类问题则最后的输出连接一个softmax函数。
总结一下为什么说回归和分类的本质是一样的，所有回归问题其实都可以变成分类问题，回归输出一个值，第一步压缩成（0,1），第二步设置一个阈值，大于阈值则为1，小于阈值则为0，这个阈值通常为0.5。
参考资料：https://www.zhihu.com/question/21329754/answer/17901883

####144.Saddle point
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-144.jpg?raw=true)
鞍点
什么是鞍点
在多维空间里，一个点是一个变量的极小值点而另一个变量的极大值点，这个点就是鞍点。
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/144_1.png?raw=true)
鞍点的性质
鞍点的特点是对所有变量的偏导数都是0。
这个点是极其重要的，在机器学习里面，我们的学习本质上是学习模型里的很多个w值，通过对w值的调整来使损失函数最小，损失函数由w来决定所以通常表示为l(w)，w值的变化是朝着损失函数l值减小的方向在进行，这个过程就叫做梯度下降。而出现的一个很大问题就是经常会下降到这个鞍点，就是各个方向的偏导数都是0，这就导致了模型没法提升。
局部最优点几乎都是鞍点
另外，常说的局部最优点大部分都是鞍点，在神经网络中参数量巨大，大家印象中的局部最优点是凹下去的点，如下图：
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/144_2.png?raw=true)
实质上不可能的，要满足这个凹下去的点要求损失函数l对每个权值w的偏导数都是0且是极小值点。每个偏导数为0的点要么是极大值点，要么是极小值点，大数定理，理论来上，两种取值情况的概率应该都是0.5。
假如一个网络的参数值都是一百万个，所以要满足这个条件要求0.5的一百万次方，几率太小了，因此，局部最优点基本不可能。

####150.Sensitive also called recall
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-150.jpg?raw=true)
召回率（查全率）
什么是召回率
召回率也就查全率，这个概念衍生自混合矩阵。
混合矩阵定义了二分类问题中的四种结果，分别是：
TP(True Positive)：预测值为真，真实值为真。
FN(False Negative)：预测值为假，真实值为真。
FP(False Positive)：预测值为真，真实值为假。
TN(True Negative)：预测值为假，真实值为假。
召回率的公式
其中召回率=TP/(TP+FN)
可以这么理解，等式下边是指所有的真实值为真的数量，等式上边是预测值为真，真实值也为真的数量。
信息检索
这个概念其实要一个真实贴近的例子才能比较好的理解，信息检索就是一个很好的适用这个概念的例子。
查全率在检索系统中：检出的相关文献与全部相关文献的百分比，衡量检索的覆盖率。
另外还有查准率。
TP/(TP+FP)：在你预测为1的样本中实际为1的概率。
查准率在检索系统中：检出的相关文献与检出的全部文献的百分比，衡量检索的信噪比。
####168.C，inverse of regularization strength
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-168.jpg?raw=true)
C，正则化系数α的倒数
什么是C
C=1/α
这里α表示的是对损失函数中正则化项的系数，这个系数是用来控制正则化在损失函数中的重要性。
为什么用C而不直接用α
取C来替代α的原因是，想要非线性的提高，举一个具体的例子当α从1到2，数值变化了1个单位，只能取得一个单位的提高，而C从0.5到0.1，数值变化0.4，但却对α取得了9.5单位的提高。
另外值得注意的是当用C来替代α对惩罚项的控制的时候，是反着的，C越小表示惩罚项占比越大，要求模型复杂度越低。
####186.The effect of feature scaling gradient descent
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-186.jpg?raw=true)
标准化输入数据
什么是标准化
对输入数据去标准化，让所有输入数据都在(0,1)范围。
为什么要做标准化
这个也是一个非常有深度的问题，为什么要对原始数据进行标准化，为什么进行了标准化可以加快收敛速度。
这个问题其实要从两方面来回答。
1. L2惩罚项倾向于惩罚数值大的w。由于惩罚项的存在，如果是用的L2惩罚项，模型更倾向于惩罚数值大的w，然而有些数据集训练的结果就是倾向于一些权值要更大才合理，而这些权值却因为L2惩罚项被强行降下去了，就变得不那么合理。例如输入的数据集包含两个特征，要预测房屋的价格和面积以及房间数的关系，对这两个特征连接的神经元w1、w2，应该是w2要比我要更大一些才合理，然而由于惩罚项L2的存在导致w2被强行降的更低。
2. 标准化能够提高收敛速度。这个是网上很多人都提过的，标准化确实能够提高收敛速度，下图就能直观的感觉到。![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/186_1.png?raw=true)，如果我们在左下角的小点点，而最优点在中间位置，没有经过标准化，则要向左移动大单位，而向上移动小单位，才能最快的速度到中间位置，而当前梯度的方向往往不是这么理想的，经常走偏，如果使用标准化则能解决这个问题，到底为什么标注化能解决这个问题，涉及到了一个矩阵的特征根能够决定训练收敛速度。具体解答可以参考下面这篇文章。
https://www.quora.com/Why-is-the-Speed-Of-Convergence-of-gradient-descent-depends-on-the-maximal-and-minimal-eigenvalues-of-A-in-solving-AX-b-through-least-squares/answer/Prasoon-Goyal
####192.Thresholding categorical feature variance
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-192.jpg?raw=true)
通过设置阈值过滤类别型变量
什么是通过设置阈值过滤类别型变量
对于类别变量，其具有一些属性，例如熵或者说IV或者说方差。
通过设置一个阈值，当高于阈值时则保留，低于则扔掉。

对卡片的翻译
1. 一个特征含有的信息量跟其方差有很大关系，其方差越大，代表这个特征含有更多信息量。
2. 在二项分布中，var(x)=p*(1-p)。
3. 设定一个阈值(Threshold)，当特征的方差低于这个阈值，就把该特征扔掉。

####198.TPR
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-198.jpg?raw=true)
TPR
什么是TPR
这个指标也是来自于混合矩阵。
TPR=TP/(TP+FN)，TPR也叫查全率
与之对应的有负样本中的错判率（假警报率）
FPR=FP/(FP+TN)
其中：
TP(True Positive)：预测值为真，真实值为真。
FN(False Negative)：预测值为假，真实值为真。
FP(False Positive)：预测值为真，真实值为假。
TN(True Negative)：预测值为假，真实值为假。
####216.When can we delete observations with misiing values
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-216.jpg?raw=true)
什么时候可以扔掉缺失值

只有在迫不得已的情况下我们才会删掉缺失值。然而，在某些时候删掉缺失值是完全可以接受的，例如，缺失值的丢失完全是我随机导致的或是由于其他特征导致的。如果不是某些特殊情况导致的数值缺失，选择直接去删除掉缺失值是不合理的。
在另外一张卡片，我遇到了一个相似的问题，那么主要在那张卡片来详细回答。
####228.Common optimizers for neural nets
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-228.jpg?raw=true)
优化器
神经网络梯度下降的四种常用优化器
几个传统的用于神经网络梯度下降的优化器：
1. SGD随机梯度下降
2. SGD-momentum，带momentum的随机梯度下降，momentum指的是动量，这个想法源自于物理的思想。
3. RMSProp，每一个参数w的梯度很过去几次梯度都关联，相当于进化版的滑动平均数。
4. Adam，Adam是目前广受欢迎的优化器，其特点就是综合了RMSProp和momentum，把两者结合了起来即为Adam。

####231.Conditional probability
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-231.jpg?raw=true)
条件概率
什么是条件概率
P(A∩B)=P(A)*P(B|A)
条件概率是指事件A在另外一个事件B已经发生条件下的发生概率。条件概率表示为：P（A|B），读作“在B的条件下A的概率”。条件概率可以用决策树进行计算。条件概率的谬论是假设 P(A|B) 大致等于 P(B|A)。数学家John Allen Paulos 在他的《数学盲》一书中指出医生、律师以及其他受过很好教育的非统计学家经常会犯这样的错误。这种错误可以通过用实数而不是概率来描述数据的方法来避免。
####234.Confusion matrix
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-234.jpg?raw=true)
混合矩阵
什么是混合矩阵
混合矩阵，也称混淆矩阵。
通过对模型输出的预测值和真实值的比较情况，做出来的一个矩阵。
混合矩阵衍生出来的指标
通过混合矩阵能衍生出很多评价模型的指标。例如查准率、查全率、ROC、AUC、KS。
####240.Cumulative distribution function
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-240.jpg?raw=true)
累积分布函数
什么是累积分布函数
F(x)=P(X<=x)
当x趋向于无穷大时，F(x)接近于1。
####246.Almost everywhere
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-246.jpg?raw=true)
几乎成立
概念
这个概念来自于数学分析的一个分支测量理论，当一个公式或属性在几乎所有地方都成立，只在部分边缘线上没有成立，则称这一性质为almost everywhere。
####252.Does K-NN learn
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-252.jpg?raw=true)
KNN算法能学习吗
KNN是解决聚类问题的一个常用算法。
这里面提到的是KNN把相似的样本聚集在一起，但是并没有进行学习。因为KNN算法模型背后根本没有参数值w。
####258.Early stopping
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-258.jpg?raw=true)
提前停止
在神经网络进行训练的时候，为了防止过度训练形成过拟合，一种方法是在验证集的损失函数要上升的时候，也就是验证集损失函数达到极小值点的时候，提前停止训练，这个方法就是early stopping。
其实这个方法也有很多能改进的地方，因为很有可能验证集的损失函数经过的是一个极小值点，后面立马又能下降了，而这个时候停止训练，是不合适的。
####264.Ensemble methods
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-264.jpg?raw=true)
集成方法
什么是集成方法
集成方法就是由多个基学习器组成的强学习器，根据组成方式的不同，可以分出多种集成方法
bagging和boosting
传统的集成方法包含两种：bagging和boosting。
bagging是指多个基分类器对来自同一训练集的不同子集进行学习，得到学习后的多个基分类器，在对新样本进行预测的时候是多个基分类器都对预测样本进行预测，把预测结果进行投票选择，最后的结果是票数最多的那个，例如共有10个基分类器，对某个样本进行预测，有7个的预测结果都为1，有3个的预测结果为0，则最后预测结果为0，这个就是bagging集成学习方法。
boosting是指提升。boosting类的算法最关键的一点就是每个基学习器是前后关联的，后一个基本学习器是通过提升前一个基学习器来得到，最后的结果是多个关联的基学习器的加权统一而得到。
####270.Extrema
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-270.jpg?raw=true)
极值点
从整个函数的角度看分为全局极值点和局部极值点。
从具体的角度看分为极大值点和极小值点。
####276.Feedforward neural network
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-278.jpg?raw=true)
前向传播神经网络
这个应该是一个很古老的叫法，在很久以前，最早制作出来就是多个神经元链接在一起，那个时候还没出现BP算法来更新权值进行学习。
这个就是当时没有BP算法的神经网络的名字。

####282.Function
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-282.jpg?raw=true)
函数
一个非常基础的概念，却有着非常深刻的应用。
一个输入值x，经过函数f，得到y。
x的取值范围称为定义域，y的取值范围称为值域。
####288.Gradient clipping
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-288.jpg?raw=true)
梯度峭壁
什么是梯度峭壁
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/wdw%E5%BC%95%E7%94%A8%E5%9B%BE%E7%89%87/288_1.png?raw=true)
损失函数的一个悬崖峭壁，意思是最优值点附近的梯度过大，极有可能不小心就跳过了这个最优点。
解决梯度峭壁的一些思路
解决方法是设置一个阈值，当梯度大于阈值的时候，压缩梯度。
####294.Greedy algorithm
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-294.jpg?raw=true)
贪心算法
什么是贪心算法
贪心算法是一种思想，每次都选当前能对结果最优的选择
贪心算法的一些特性
1. 只考虑当前情况最好的选择。
2. 贪心算法不是一个具体的算法，而是一种思想，这种思想广泛的应用在个种算法当中，但是贪心算法也会带来很不好的地方，就是容易进入局部最佳，这是因为缺乏对全局的认识导致的。
####300.Hadamard product
edit by wdw
![](https://github.com/6studentsfromsspku/sspku-300-concepts/blob/master/%E5%8D%A1%E7%89%87%E9%9B%86%E5%90%88/%E5%8D%A1%E7%89%87-300.jpg?raw=true)
矩阵乘法
什么是矩阵乘法
就是两个矩阵相乘。
矩阵乘法的一些性质
具有以下性质
性质1：  $ \frac{1}{\sqrt{n}} Hn$为正交方阵，所谓正交矩阵指它的任意两行（或两列）都是正交的。并且行列式为+1或-1。
性质2:任意一行（列）的所有元素的平方和等于方阵的阶数。即：设A为n阶由+1和-1元素构成的方阵，若AA‘=nI（这里A’为A的转置，I为单位方阵）。
性质3：Hadamard矩阵的阶数都是2或者是4的倍数。
性质4：若M为n阶实方阵，若M的所有元素的绝对值均小于1，则M的行列式$\leq n ^ (0.5n) $，当且仅当M为哈达玛矩阵时取等。（此结论由哈达玛不等式得出）
