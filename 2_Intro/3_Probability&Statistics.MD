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

混合模型是组合简单概率分布来生成更复杂模型的一种简单策略。

由混合模型的概念出发，我们能够提前了解几个重要概念。

- 潜变量：我们不能直接观测到的随机变量
- 高斯混和模型