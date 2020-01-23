# 概率论

本节目标：回忆可能用到的概率论相关知识点

## 1.随机变量

## 2.概率分布

### 2.1离散型：概率质量函数

### 2.2连续型：概率密度函数

## 3.边缘概率

## 4.条件概率

### 4.1条件概率的乘法法则

## 5.独立性和条件独立性

## 6.数学特征

### 6.1期望

### 6.2方差

### 6.3协方差

### 6.4相关系数

## 7.常用概率分布

### 7.1伯努利分布

$$
\begin{aligned} P(\mathrm{x}=1) &=\phi \\ P(\mathrm{x}=0)&= 1-\phi \\ P(\mathrm{x}=x)&=\phi^{x}(1-\phi)^{1-x} \\ \mathbb{E}_{\mathrm{x}}[\mathrm{x}]&=\phi \\ \operatorname{Var}_{\mathrm{x}}(\mathrm{x})&=\phi(1-\phi) \end{aligned}
$$

### 7.2几何分布/范畴分布

### 7.3正态分布/高斯分布

$$
\mathcal{N}\left(x ; \mu, \sigma^{2}\right)=\sqrt{\frac{1}{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)
$$

#### 7.3.1一般参数

在正态分布中的常用参数是前两个：均值和方差。当我们需要经常对不同参数下的概率密度函数求值时，一种更高效的参数化分布方式是使用参数$\beta$
$$
\mathcal{N}\left(x ; \mu, \beta^{-1}\right)=\sqrt{\frac{\beta}{2 \pi}} \exp \left(-\frac{1}{2} \beta(x-\mu)^{2}\right)
$$

- 均值$\mu$
- 方差$\sigma^2$
- 精度$\beta$:方差的倒数

采用正态分布在很多应用中是个正确的选择。由于缺乏先验知识，正态分布是一个比较好的默认开始，主要由于以下两个原因：

- 中心极限定理说明很多独立随机变量的和近似服从正态分布。这意味着在实际中，很多复杂系统都可以被成功建模成正态分布的噪声，即使得系统可以分解成一些更加结构化的成分。

- 在具有相同方差的所有可能的概率分布中，正态分布在实数上具有最大的不确定性。因此，我们可以认为正态分布是对模型加入的先验知识量最少的分布。

#### 7.3.2多维正态分布

正态分布可以推广至$\R^n$空间，这种情况下我们称为多维正态分布，他的参数是一个正定矩阵。
$$
\mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}, \mathbf{\Sigma})=\sqrt{\frac{1}{(2 \pi)^{n} \operatorname{det}(\boldsymbol{\Sigma})}} \exp \left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

此时$\mu$是一个向量值，参数$\Sigma$给出分布的协方差矩阵。当我们需要经常对不同参数下的概率密度函数求值时，一种更高效的参数化分布方式是使用精度矩阵$\beta$
$$
\mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}, \boldsymbol{\beta}^{-1}\right)=\sqrt{\frac{\operatorname{det}(\boldsymbol{\beta})}{(2 \pi)^{n}}} \exp \left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\beta}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

我们常把协方差矩阵固定成一个对角阵。

一个更简单的版本是各向同性高斯分布，它的协方差是一个标量乘以单位阵。

### 7.4指数分布和Laplace分布

在深度学习中，常需要一个在$x = 0$点取得边界点的分布。我们使用指数分布：
$$
p(x ; \lambda)=\lambda \mathbf{1}_{x \geqslant 0} \exp (-\lambda x)
$$

一个联系紧密的概率分布是Laplace分布，它允许我们在任何一个$\mu$处设置概率质量的峰值。

$$
\text { Laplace }(x ; \mu, \gamma)=\frac{1}{2 \gamma} \exp \left(-\frac{|x-\mu|}{\gamma}\right)
$$

### 7.5Dirac分布

其概率分布都集中在一个点上，运用$\delta$函数实现。

$$
p(x)=\delta(x-\mu)
$$

可平移使用。

Dirac分布常作为经验分布的一个组成部分。
$$
\hat{p}(\boldsymbol{x})=\frac{1}{m} \sum_{i=1}^{m} \delta\left(\boldsymbol{x}-\boldsymbol{x}^{(i)}\right)
$$

经验分布将概率密度的$\frac{1}{m}$分给$n$个点，这些点是给定的数据集或者采样的集合。

在我们在训练集上训练模型时，可以认为从这个训练集上得到的经验分布指明了采样来源的分布。

另一个重要观点是，他是训练数据的似然最大的那个函数。

### 分布的混合

一般情况下，常用多个简单分布构造一个混合分布。
$$
P(\mathrm{x})=\sum_{i} P(\mathrm{c}=i) P(\mathrm{x} | \mathrm{c}=i)
$$

混合模型是组合简单概率分布来生成更复杂模型的一种简单策略。

由混合模型的概念出发，我们能够提前了解几个重要概念。

- 潜变量：我们不能直接观测到的随机变量
- 高斯混和模型：万能近似器，每个组件是高斯分布
  - 先验概率
  - 后验概率

`任何平滑的概率密度都可以用足够多组件的高斯混合模型以任意精度逼近。`

## 8.常用函数

### 8.1 Logistic Sigmoid函数

$$
\sigma(x)=\frac{1}{1+\exp (-x)}
$$

其常用来生成Bernoulli分布的参数$\phi$

函数值域为$(0,1)$

其在变量取非常大的绝对值时会出现“饱和”现象：函数变得很平，对于变化不敏感。

### 8.2 Softplus函数

$$
\zeta(x)=\log (1+\exp (x))
$$

Softplus函数常用来产生正态分布的$\beta$和$\sigma$参数。(因为他的值域是$(0,+\infty)$)

其名称“软化”是因为其是另一个函数的平滑形式：

$$
x^{+}=\max (0, x)
$$

一些常用的性质如下所示。

$$
\begin{aligned} \sigma(x) &=\frac{\exp (x)}{\exp (x)+\exp (0)} \\ \frac{d}{d x} \sigma(x) &=\sigma(x)(1-\sigma(x)) \\ 1-\sigma(x) &=\sigma(-x) \\ \log \sigma(x) &=-\zeta(-x) \\ \frac{d}{d x} \zeta(x) &=\sigma(x) \\ \forall x \in(0,1), \sigma^{-1}(x) &=\log \left(\frac{x}{1-x}\right) \\ \forall x>0, \zeta^{-1}(x)&=\log (\exp (x)-1) & \\ \zeta(x) &=\int_{-\infty}^{x} \sigma(y) d y \\ \zeta(x) &-\zeta(-x)=x \end{aligned}
$$

## 9.贝叶斯规则

## 10.连续性变量的随机细节

PS：这一方面需要运用测度论的相关知识

- 零测度：非常微小的点集
  - 几乎处处(almost everywhere)：如果某个性质几乎出处成立，那么它在整个空间中除了一个测度为零的集合意外都是成立的。
- 对于空间形变的处理
  - 高维空间中的微分运算：Jacobian矩阵

## 11.信息论

在机器学习中，我们主要使用信息论的一些关键思想来描述概率分布或者量化概率分布之间的相似性。

- 香农熵
  - 微分熵
- KL散度
- 交叉熵

## 12.结构化概率模型

机器学习的算法经常会涉及在非常多的随机变量上的概率分布。通常，这些概率分布涉及的直接相互作用都是介于非常少的变量之间的。使用单个函数来描述整个联合概率分布是非常低效的（无论是计算上还是统计上）。

更加高效的做法是：把概率分布分解成许多因子的乘积形式，而不是使用单一的函数来表示概率分布。

这种分解可以极大地减少用来描述一个分布的参数数量。每个因子使用的参数数目是其变量数目的指数倍。这意味着，如果我们能够找到一种使每个因子分布具有更少变量的分解方法，就能极大地降低表示联合分布的成本。

可以用图来描述这种分解。这里我们使用的是图论中的“图”的概念:由一些可以通过边互相连接的顶点的集合构成。当用图来表示这种概率分布的分解时，我们把它称为结构化概率模型(structured probabilistic model)或者图模型(graphical model)

有两种模型来描述这种分解：有向图和无向图。

### 12.1 有向图

有向（directed）模型使用带有有向边的图，它们用条件概率分布来表示分解。特别地，有向模型对于分布中的每一个随机变量$x_i$；都包含着一个影响因子，这个组成$x_i$条件概率的影响因子被称为$x_i$的父节点，记为$P a_{\mathcal{G}}\left(\mathrm{x}_{i}\right)$

$$
p(\mathbf{x})=\prod_{i} p\left(\mathbf{x}_{i} | P a_{\mathcal{G}}\left(\mathbf{x}_{i}\right)\right)
$$

### 12.2 无向图

无向（undirected）模型使用带有无向边的图，们将分解表示成一组函数:不像有向模型那样，这些函数通常不是任何类型的概率分布。$\mathcal{G}$中任何满足两两之间有边连接的顶点的集合被称为团。无向模型中的每个团$\mathcal{C}$都伴随着一个因子$\phi^{i}(\mathcal{C}^{i})$。这些因子仅仅是函数并不是概率分布。每个因子的输出都必须是非负的，但是并没有像概率分布中那样要求因子的和或者积分为1

随机变量的联合概率与所有这些因子的乘积成比例 proportional）这意味着因子的值越大，则可能性越大。当然，不能保证这种乘积的求和为1。所以我们需要除以一个归一化常数$Z$来得到归一化的概率分布，归一化常数$Z$被定义为$\phi$函数乘积的所有状态的求和或积分。

$$
p(\mathbf{x})=\frac{1}{Z} \prod_{i} \phi^{(i)}\left(\mathcal{C}^{(i)}\right)
$$
