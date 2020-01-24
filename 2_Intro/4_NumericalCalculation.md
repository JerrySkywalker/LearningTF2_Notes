# 数值计算

## 1.上溢和下溢

连续数学在数字计算机上的根本困难是：我们需要用有限多位来表达无穷位数字。在这种情况下，一般要进行舍入误差的操作。

- 下溢
- 上溢

为了对于上溢和下溢进行一定的稳定，一般使用Softplus函数或者log Softplus函数。

大多是库工具已经进行了数值稳定操作，但平常使用中仍需十分小心类似问题的出现。

## 2.病态条件

条件数：函数相对于输入的微小变化而变化的快慢程度。对一个函数：
$$
f(x)=A^{-1} x
$$
当
$$
A \in \mathbb{R}^{n \times n}
$$
具有特征值分解时，规定其条件数为：
$$
\max _{i, j}\left|\frac{\lambda_{i}}{\lambda_{j}}\right|
$$

- 大小：最大特征值和最小特征值之比
- 是矩阵的固有特性

## 3.基于梯度的优化方法

优化问题通常用最小化$f(x)$来代替。最大化时则取$-f(x)$。在这里，$f(x)$一般称为目标函数，也有如下别称，一般情况下可以混用：

- 目标函数(Objective Function)
- 准则(Criterion)
- 代价函数(Cost Function)
- 损失函数(Loss Function)
- 误差函数(Error Function)

为了求解函数的最低点，对一元函数一般采用求导的方法，高维函数采用求梯度的方法。计算某一个函数的方向导数：
$$
\begin{array}{c}{\min _{u, u^{\top} u=1} u^{\top} \nabla_{x} f(x)} \\ {=\min _{u, u^{\top} u=1}\|u\|_{2}\left\|\nabla_{x} f(x)\right\|_{2} \cos \theta}\end{array}
$$

我们希望找到梯度下降最快的方向，并以此多次迭代操作。由此引出最速下降法/梯度下降法。

梯度下降法求得的新点位置是
$$
x^{\prime}=x-\epsilon \nabla_{x} f(x)
$$

其中$\epsilon$称为学习率。设置合适的学习率是非常关键的：过小的学习率使得训练时间过长，过大的学习率将使得梯度下降法无法找到最低点。

该方法的一般概念可以推广到离散空间--爬山算法(Hill Climbing Algorithm)

## 4.梯度之上：Jacobian和Hessian矩阵

有时，我们需要计算输入输出都是都是向量的函数的偏导数。包含所有这样偏导数的矩阵称为雅各布(Jacobian)矩阵。

对$f: \mathbb{R}^{m} \rightarrow \mathbb{R}^{n}$，我们规定他的Jacobian矩阵为$J \in \mathbb{R}^{n \times m}$

$$
J_{i, j}=\frac{\partial}{\partial x_{j}} f(x)_{i}
$$

相对应的，包含高维函数全部的二阶倒数的矩阵称为Hessian矩阵$H(f)(x)$

$$
H(f)(x)_{i, j}=\frac{\partial^{2}}{\partial x_{i} \partial x_{j}} f(x)
$$

有两点需要注意：

- Hessian矩阵等价与梯度的Jacobian矩阵。
- 由于微分算子对二阶偏导连续的点可交换，而在深度学习中遇到的函数一般也是如此，所以一般情况下，Hessian矩阵一般是实对称的，

$$
\frac{\partial^{2}}{\partial x_{i} \partial x_{j}} f(\boldsymbol{x})=\frac{\partial^{2}}{\partial x_{j} \partial x_{i}} f(\boldsymbol{x})
$$