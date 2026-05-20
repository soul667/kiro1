# EE5346 - 自主机器人导航（2026春季）
# 作业 #5 参考解答

## 1. 最大似然估计（Maximum Likelihood Estimation, MLE）

已知测量模型：

\[
z_k = x + n_k
\]

其中：

- \(x\)：待估计的机器人位置（未知常数）
- \(n_k \sim \mathcal{N}(0,\sigma^2)\)：高斯噪声
- \(z_k\)：第 \(k\) 次测量值

共有 \(N\) 个独立测量：

\[
Z = [z_1,z_2,\cdots,z_N]^T
\]

由于各测量独立，因此：

\[
p(Z|x)=\prod_{k=1}^{N} p(z_k|x)
\]

---

## 1.1 单个测量的概率密度函数

由高斯分布可知：

\[
z_k \sim \mathcal{N}(x,\sigma^2)
\]

因此：

\[
p(z_k|x)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(z_k-x)^2}{2\sigma^2}\right]
\]

---

## 1.2 似然函数 \(L(x)=p(Z|x)\)

由于所有测量独立：

\[
L(x)=p(Z|x)
=\prod_{k=1}^{N}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(z_k-x)^2}{2\sigma^2}\right]
\]

整理常数项：

\[
L(x)=
\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^N
\exp\left[-\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
\right]
\]

因此：

\[
\boxed{
L(x)=
\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^N
\exp\left[-\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
\right]
}
\]

---

## 1.3 负对数似然函数

定义：

\[
\mathcal{L}(x)=-\ln L(x)
\]

代入上式：

\[
\mathcal{L}(x)
=-\ln
\left[
\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^N
\exp\left(-\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
\right)
\right]
\]

利用对数性质：

\[
\ln(ab)=\ln a+\ln b
\]

得到：

\[
\mathcal{L}(x)
=-N\ln\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)
+\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
\]

进一步整理：

\[
\boxed{
\mathcal{L}(x)
=\frac{N}{2}\ln(2\pi\sigma^2)
+\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
}
\]

其中第一项与 \(x\) 无关，因此：

\[
\arg\min_x \mathcal{L}(x)
\equiv
\arg\min_x
\sum_{k=1}^{N}(z_k-x)^2
\]

即 MLE 等价于最小二乘问题。

---

## 1.4 求最大似然估计 \(x^*\)

对目标函数求导：

\[
J(x)=\sum_{k=1}^{N}(z_k-x)^2
\]

对 \(x\) 求导：

\[
\frac{dJ}{dx}
=\sum_{k=1}^{N}2(x-z_k)
\]

令导数为零：

\[
\sum_{k=1}^{N}2(x-z_k)=0
\]

\[
2Nx-2\sum_{k=1}^{N}z_k=0
\]

\[
Nx=\sum_{k=1}^{N}z_k
\]

因此：

\[
\boxed{
x^*=\frac{1}{N}\sum_{k=1}^{N}z_k
}
\]

即：

> 高斯噪声下，机器人位置的 MLE 等于所有测量值的样本均值。

---

## 1.5 数值计算

已知：

\[
N=3
\]

\[
z_1=3,\quad z_2=4,\quad z_3=7
\]

代入公式：

\[
x^*=\frac{3+4+7}{3}
\]

\[
\boxed{
x^*=\frac{14}{3}\approx 4.667
}
\]

---

# 2. EKF 预测均值与协方差推导

目标证明：

\[
\tilde{\mathbf{x}}_k=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
\]

\[
\tilde{\mathbf{P}}_k
=\mathbf{F}\hat{\mathbf{P}}_{k-1}\mathbf{F}^T+\mathbf{R}_k
\]

其中：

- \(\hat{\mathbf{x}}_{k-1}\)：上一时刻后验估计均值
- \(\hat{\mathbf{P}}_{k-1}\)：上一时刻后验协方差
- \(f(\cdot)\)：非线性运动模型
- \(\mathbf{F}\)：运动模型关于状态的一阶雅可比
- \(\mathbf{R}_k\)：过程噪声协方差

---

## 2.1 非线性状态方程

系统模型：

\[
\mathbf{x}_k=f(\mathbf{x}_{k-1},\mathbf{u}_k)+\mathbf{w}_k
\]

其中：

\[
\mathbf{w}_k\sim\mathcal{N}(0,\mathbf{R}_k)
\]

已知：

\[
\mathbf{x}_{k-1}\sim
\mathcal{N}(\hat{\mathbf{x}}_{k-1},\hat{\mathbf{P}}_{k-1})
\]

由于 \(f\) 为非线性函数，因此 EKF 采用一阶泰勒展开。

---

## 2.2 一阶泰勒展开

在估计点 \(\hat{\mathbf{x}}_{k-1}\) 处展开：

\[
f(\mathbf{x}_{k-1},\mathbf{u}_k)
\approx
f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
+\mathbf{F}(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})
\]

其中：

\[
\boxed{
\mathbf{F}
=\left.
\frac{\partial f}{\partial \mathbf{x}}
\right|_{\hat{\mathbf{x}}_{k-1},\mathbf{u}_k}
}
\]

因此：

\[
\mathbf{x}_k
\approx
f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
+\mathbf{F}(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})
+\mathbf{w}_k
\]

---

## 2.3 预测均值推导

均值定义：

\[
\tilde{\mathbf{x}}_k=E[\mathbf{x}_k]
\]

代入线性化表达式：

\[
\tilde{\mathbf{x}}_k
=E\Big[
 f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
+\mathbf{F}(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})
+\mathbf{w}_k
\Big]
\]

利用期望线性性质：

\[
\tilde{\mathbf{x}}_k
=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
+\mathbf{F}E[\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}]
+E[\mathbf{w}_k]
\]

由于：

\[
E[\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}]=0
\]

且：

\[
E[\mathbf{w}_k]=0
\]

因此：

\[
\boxed{
\tilde{\mathbf{x}}_k
=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
}
\]

---

## 2.4 预测协方差推导

协方差定义：

\[
\tilde{\mathbf{P}}_k
=E\left[
(\mathbf{x}_k-\tilde{\mathbf{x}}_k)
(\mathbf{x}_k-\tilde{\mathbf{x}}_k)^T
\right]
\]

由上节结果：

\[
\tilde{\mathbf{x}}_k=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
\]

因此：

\[
\mathbf{x}_k-\tilde{\mathbf{x}}_k
=\mathbf{F}(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})+\mathbf{w}_k
\]

代入协方差定义：

\[
\tilde{\mathbf{P}}_k
=E\Big[
(\mathbf{F}\delta\mathbf{x}_{k-1}+\mathbf{w}_k)
(\mathbf{F}\delta\mathbf{x}_{k-1}+\mathbf{w}_k)^T
\Big]
\]

其中：

\[
\delta\mathbf{x}_{k-1}=\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}
\]

展开：

\[
\tilde{\mathbf{P}}_k
=
E\Big[
\mathbf{F}\delta\mathbf{x}_{k-1}\delta\mathbf{x}_{k-1}^T\mathbf{F}^T
+\mathbf{F}\delta\mathbf{x}_{k-1}\mathbf{w}_k^T
+\mathbf{w}_k\delta\mathbf{x}_{k-1}^T\mathbf{F}^T
+\mathbf{w}_k\mathbf{w}_k^T
\Big]
\]

由于状态误差与过程噪声独立：

\[
E[\delta\mathbf{x}_{k-1}\mathbf{w}_k^T]=0
\]

因此交叉项消失：

\[
\tilde{\mathbf{P}}_k
=
\mathbf{F}
E[\delta\mathbf{x}_{k-1}\delta\mathbf{x}_{k-1}^T]
\mathbf{F}^T
+E[\mathbf{w}_k\mathbf{w}_k^T]
\]

根据协方差定义：

\[
E[\delta\mathbf{x}_{k-1}\delta\mathbf{x}_{k-1}^T]
=\hat{\mathbf{P}}_{k-1}
\]

且：

\[
E[\mathbf{w}_k\mathbf{w}_k^T]=\mathbf{R}_k
\]

因此得到：

\[
\boxed{
\tilde{\mathbf{P}}_k
=\mathbf{F}\hat{\mathbf{P}}_{k-1}\mathbf{F}^T+\mathbf{R}_k
}
\]

这就是 EKF 的预测协方差方程。

---

# 3. EKF 雅可比矩阵推导

状态向量：

\[
\mathbf{x}_k=
\begin{bmatrix}
x_k\\
y_k\\
\theta_k
\end{bmatrix}
\]

控制输入：

\[
\mathbf{u}_k=
\begin{bmatrix}
v_k\\
\omega_k
\end{bmatrix}
\]

---

# 3.1 运动模型

题目给定：

\[
\mathbf{x}_k
=
\mathbf{x}_{k-1}
+
\Delta t
\begin{bmatrix}
\cos\theta_{k-1} & 0\\
\sin\theta_{k-1} & 0\\
0 & 1
\end{bmatrix}
(\mathbf{u}_k+\mathbf{w}_k)
\]

忽略噪声项用于求雅可比：

\[
\mathbf{x}_k=f(\mathbf{x}_{k-1},\mathbf{u}_k)
\]

展开得到：

\[
\begin{aligned}
x_k
&=x_{k-1}+\Delta t\,v_k\cos\theta_{k-1}
\\[4pt]
y_k
&=y_{k-1}+\Delta t\,v_k\sin\theta_{k-1}
\\[4pt]
\theta_k
&=\theta_{k-1}+\Delta t\,\omega_k
\end{aligned}
\]

---

# 3.2 状态转移雅可比矩阵 \(\mathbf{F}\)

定义：

\[
\mathbf{F}
=
\frac{\partial f}{\partial \mathbf{x}}
\]

即：

\[
\mathbf{F}
=
\begin{bmatrix}
\dfrac{\partial x_k}{\partial x_{k-1}} &
\dfrac{\partial x_k}{\partial y_{k-1}} &
\dfrac{\partial x_k}{\partial \theta_{k-1}}
\\[10pt]
\dfrac{\partial y_k}{\partial x_{k-1}} &
\dfrac{\partial y_k}{\partial y_{k-1}} &
\dfrac{\partial y_k}{\partial \theta_{k-1}}
\\[10pt]
\dfrac{\partial \theta_k}{\partial x_{k-1}} &
\dfrac{\partial \theta_k}{\partial y_{k-1}} &
\dfrac{\partial \theta_k}{\partial \theta_{k-1}}
\end{bmatrix}
\]

逐项求导：

---

## 第一行

\[
x_k=x_{k-1}+\Delta t\,v_k\cos\theta_{k-1}
\]

因此：

\[
\frac{\partial x_k}{\partial x_{k-1}}=1
\]

\[
\frac{\partial x_k}{\partial y_{k-1}}=0
\]

\[
\frac{\partial x_k}{\partial \theta_{k-1}}
=-\Delta t\,v_k\sin\theta_{k-1}
\]

---

## 第二行

\[
y_k=y_{k-1}+\Delta t\,v_k\sin\theta_{k-1}
\]

因此：

\[
\frac{\partial y_k}{\partial x_{k-1}}=0
\]

\[
\frac{\partial y_k}{\partial y_{k-1}}=1
\]

\[
\frac{\partial y_k}{\partial \theta_{k-1}}
=\Delta t\,v_k\cos\theta_{k-1}
\]

---

## 第三行

\[
\theta_k=\theta_{k-1}+\Delta t\,\omega_k
\]

因此：

\[
\frac{\partial \theta_k}{\partial x_{k-1}}=0
\]

\[
\frac{\partial \theta_k}{\partial y_{k-1}}=0
\]

\[
\frac{\partial \theta_k}{\partial \theta_{k-1}}=1
\]

---

## 最终结果

因此状态转移雅可比矩阵为：

\[
\boxed{
\mathbf{F}
=
\begin{bmatrix}
1 & 0 & -\Delta t\,v_k\sin\theta_{k-1}
\\[6pt]
0 & 1 & \Delta t\,v_k\cos\theta_{k-1}
\\[6pt]
0 & 0 & 1
\end{bmatrix}
}
\]

---

# 3.3 测量模型

定义：

\[
\mathbf{z}_k^l
=h(\mathbf{x}_k,\mathbf{y}^l)
=
\begin{bmatrix}
r\\
\phi
\end{bmatrix}
\]

其中：

\[
r=
\sqrt{(x_l-x_k-d\cos\theta_k)^2
+(y_l-y_k-d\sin\theta_k)^2}
\]

\[
\phi=
\operatorname{atan2}
(y_l-y_k-d\sin\theta_k,
 x_l-x_k-d\cos\theta_k)
-\theta_k
\]

定义中间变量：

\[
\Delta x=x_l-x_k-d\cos\theta_k
\]

\[
\Delta y=y_l-y_k-d\sin\theta_k
\]

则：

\[
r=\sqrt{\Delta x^2+\Delta y^2}
\]

\[
\phi=\operatorname{atan2}(\Delta y,\Delta x)-\theta_k
\]

---

# 3.4 测量雅可比矩阵 \(\mathbf{H}\)

定义：

\[
\mathbf{H}
=
\frac{\partial h}{\partial \mathbf{x}}
\]

即：

\[
\mathbf{H}
=
\begin{bmatrix}
\dfrac{\partial r}{\partial x_k} &
\dfrac{\partial r}{\partial y_k} &
\dfrac{\partial r}{\partial \theta_k}
\\[10pt]
\dfrac{\partial \phi}{\partial x_k} &
\dfrac{\partial \phi}{\partial y_k} &
\dfrac{\partial \phi}{\partial \theta_k}
\end{bmatrix}
\]

---

# 3.5 距离测量的偏导数

由于：

\[
r=\sqrt{\Delta x^2+\Delta y^2}
\]

因此：

\[
\frac{\partial r}{\partial x_k}
=
\frac{1}{2}(\Delta x^2+\Delta y^2)^{-1/2}(2\Delta x)(-1)
\]

即：

\[
\frac{\partial r}{\partial x_k}
=-\frac{\Delta x}{r}
\]

同理：

\[
\frac{\partial r}{\partial y_k}
=-\frac{\Delta y}{r}
\]

对于 \(\theta_k\)：

\[
\frac{\partial \Delta x}{\partial \theta_k}
=d\sin\theta_k
\]

\[
\frac{\partial \Delta y}{\partial \theta_k}
=-d\cos\theta_k
\]

因此：

\[
\frac{\partial r}{\partial \theta_k}
=
\frac{1}{r}
\left(
\Delta x\frac{\partial \Delta x}{\partial \theta_k}
+
\Delta y\frac{\partial \Delta y}{\partial \theta_k}
\right)
\]

得到：

\[
\frac{\partial r}{\partial \theta_k}
=
\frac{d}{r}
(\Delta x\sin\theta_k-\Delta y\cos\theta_k)
\]

---

# 3.6 方位角测量的偏导数

已知：

\[
\phi=\operatorname{atan2}(\Delta y,\Delta x)-\theta_k
\]

利用 atan2 的求导公式：

\[
\frac{\partial}{\partial x}\operatorname{atan2}(y,x)
=-\frac{y}{x^2+y^2}
\]

\[
\frac{\partial}{\partial y}\operatorname{atan2}(y,x)
=\frac{x}{x^2+y^2}
\]

因此：

\[
\frac{\partial \phi}{\partial x_k}
=
\frac{\Delta y}{r^2}
\]

\[
\frac{\partial \phi}{\partial y_k}
=-\frac{\Delta x}{r^2}
\]

对于 \(\theta_k\)：

\[
\frac{\partial \phi}{\partial \theta_k}
=
\frac{\partial}{\partial \theta_k}
\operatorname{atan2}(\Delta y,\Delta x)-1
\]

又因为：

\[
\frac{\partial}{\partial \theta_k}
\operatorname{atan2}(\Delta y,\Delta x)
=
\frac{1}{r^2}
\left(
\Delta x\frac{\partial \Delta y}{\partial \theta_k}
-
\Delta y\frac{\partial \Delta x}{\partial \theta_k}
\right)
\]

代入：

\[
\frac{\partial \Delta x}{\partial \theta_k}=d\sin\theta_k
\]

\[
\frac{\partial \Delta y}{\partial \theta_k}=-d\cos\theta_k
\]

得到：

\[
\frac{\partial \phi}{\partial \theta_k}
=
\frac{1}{r^2}
(-d\Delta x\cos\theta_k-d\Delta y\sin\theta_k)-1
\]

即：

\[
\frac{\partial \phi}{\partial \theta_k}
=-\frac{d}{r^2}
(\Delta x\cos\theta_k+\Delta y\sin\theta_k)-1
\]

---

# 3.7 最终测量雅可比矩阵

因此：

\[
\boxed{
\mathbf{H}
=
\begin{bmatrix}
-\dfrac{\Delta x}{r}
&
-\dfrac{\Delta y}{r}
&
\dfrac{d}{r}(\Delta x\sin\theta_k-\Delta y\cos\theta_k)
\\[12pt]
\dfrac{\Delta y}{r^2}
&
-\dfrac{\Delta x}{r^2}
&
-\dfrac{d}{r^2}(\Delta x\cos\theta_k+\Delta y\sin\theta_k)-1
\end{bmatrix}
}
\]

其中：

\[
\Delta x=x_l-x_k-d\cos\theta_k
\]

\[
\Delta y=y_l-y_k-d\sin\theta_k
\]

\[
r=\sqrt{\Delta x^2+\Delta y^2}
\]

---

# 最终答案总结

## 题 1：MLE

### 似然函数

\[
L(x)=
\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^N
\exp\left[-\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
\right]
\]

### 负对数似然

\[
\mathcal{L}(x)
=\frac{N}{2}\ln(2\pi\sigma^2)
+\frac{1}{2\sigma^2}
\sum_{k=1}^{N}(z_k-x)^2
\]

### MLE

\[
x^*=\frac{1}{N}\sum_{k=1}^{N}z_k
\]

数值结果：

\[
\boxed{x^*=14/3\approx 4.667}
\]

---

## 题 2：EKF 预测

### 预测均值

\[
\boxed{
\tilde{\mathbf{x}}_k=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
}
\]

### 预测协方差

\[
\boxed{
\tilde{\mathbf{P}}_k
=\mathbf{F}\hat{\mathbf{P}}_{k-1}\mathbf{F}^T+\mathbf{R}_k
}
\]

---

## 题 3：EKF 雅可比矩阵

### 状态转移雅可比

\[
\boxed{
\mathbf{F}
=
\begin{bmatrix}
1 & 0 & -\Delta t\,v_k\sin\theta_{k-1}
\\[6pt]
0 & 1 & \Delta t\,v_k\cos\theta_{k-1}
\\[6pt]
0 & 0 & 1
\end{bmatrix}
}
\]

### 测量雅可比

\[
\boxed{
\mathbf{H}
=
\begin{bmatrix}
-\dfrac{\Delta x}{r}
&
-\dfrac{\Delta y}{r}
&
\dfrac{d}{r}(\Delta x\sin\theta_k-\Delta y\cos\theta_k)
\\[12pt]
\dfrac{\Delta y}{r^2}
&
-\dfrac{\Delta x}{r^2}
&
-\dfrac{d}{r^2}(\Delta x\cos\theta_k+\Delta y\sin\theta_k)-1
\end{bmatrix}
}
\]

