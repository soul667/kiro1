# EE5346 - 自主机器人导航（2026 春季）

## 作业 #5：MLE 与扩展卡尔曼滤波器（最终解答）

> 截止日期：2026-06-01  
> 学分：5%

---

## 题 1：最大似然估计（MLE）

### 1.1 模型与符号

测量方程：

$$
z_k = x + n_k,\qquad n_k \sim \mathcal{N}(0,\sigma^2),\qquad k=1,2,\dots,N
$$

各测量相互独立，待估参数为常量 $x$。题目中的 $\mathcal{N}(0,\sigma)$ 中 $\sigma$ 为标准差，方差为 $\sigma^2$。

由于 $z_k\mid x \sim \mathcal{N}(x,\sigma^2)$，单测量 PDF 为：

$$
p(z_k\mid x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left[-\frac{(z_k-x)^2}{2\sigma^2}\right]
$$

### 1.2 似然函数 $L(x)=p(Z\mid x)$

测量独立 $\Rightarrow$ 联合 PDF 等于各 PDF 乘积：

$$
\boxed{\;
L(x)=\prod_{k=1}^{N}p(z_k\mid x)
=(2\pi\sigma^2)^{-N/2}\exp\!\left[-\frac{1}{2\sigma^2}\sum_{k=1}^{N}(z_k-x)^2\right]
\;}
$$

### 1.3 负对数似然 $\mathcal{L}(x)=-\ln L(x)$

取负对数并利用 $\ln(ab)=\ln a+\ln b$：

$$
\boxed{\;
\mathcal{L}(x)=\frac{N}{2}\ln(2\pi\sigma^2)+\frac{1}{2\sigma^2}\sum_{k=1}^{N}(z_k-x)^2
\;}
$$

第一项与 $x$ 无关，因此

$$
x^{*}=\arg\min_{x}\mathcal{L}(x)\;\equiv\;\arg\min_{x}\sum_{k=1}^{N}(z_k-x)^2
$$

即 **高斯噪声下的 MLE 等价于最小二乘**。

### 1.4 求解 MLE

对 $\mathcal{L}(x)$ 求导并令其为零：

$$
\frac{d\mathcal{L}}{dx}=\frac{1}{\sigma^2}\sum_{k=1}^{N}(x-z_k)=0
\;\Longrightarrow\;
Nx=\sum_{k=1}^{N}z_k
$$

二阶导 $\dfrac{d^2\mathcal{L}}{dx^2}=\dfrac{N}{\sigma^2}>0$，确认极小值。因此：

$$
\boxed{\;x^{*}=\frac{1}{N}\sum_{k=1}^{N}z_k\;}
$$

> **结论：在 i.i.d. 高斯噪声下，位置的 MLE 就是测量样本均值。**

### 1.5 数值结果

$N=3$，$z_1=3,\;z_2=4,\;z_3=7$：

$$
\boxed{\;x^{*}=\frac{3+4+7}{3}=\frac{14}{3}\approx 4.667\;}
$$

---

## 题 2：教材式 (8.32) 的推导（EKF 预测均值与协方差）

### 2.1 系统模型与已知条件

非线性运动模型（加性过程噪声形式）：

$$
\mathbf{x}_k=f(\mathbf{x}_{k-1},\mathbf{u}_k)+\mathbf{w}_k,
\qquad \mathbf{w}_k\sim\mathcal{N}(\mathbf{0},\mathbf{R}_k)
$$

其中 $\mathbf{w}_k$ 与历史状态独立。上一时刻后验：

$$
\mathbf{x}_{k-1}\sim\mathcal{N}(\hat{\mathbf{x}}_{k-1},\hat{\mathbf{P}}_{k-1})
$$

### 2.2 在估计点处一阶泰勒展开

由于 $f$ 非线性，EKF 在 $\hat{\mathbf{x}}_{k-1}$ 处线性化：

$$
f(\mathbf{x}_{k-1},\mathbf{u}_k)\approx
f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)+\mathbf{F}(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})
$$

其中

$$
\mathbf{F}=\left.\frac{\partial f}{\partial\mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1},\mathbf{u}_k}
$$

带入状态方程：

$$
\mathbf{x}_k\approx
f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)+\mathbf{F}(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})+\mathbf{w}_k
\tag{$\star$}
$$

### 2.3 预测均值 $\tilde{\mathbf{x}}_k$

由 **均值是期望** 的定义 $\tilde{\mathbf{x}}_k=E[\mathbf{x}_k]$，对 $(\star)$ 两边取期望并利用线性性：

$$
\tilde{\mathbf{x}}_k=
f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)
+\mathbf{F}\underbrace{E[\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}]}_{=\mathbf{0}}
+\underbrace{E[\mathbf{w}_k]}_{=\mathbf{0}}
$$

故：

$$
\boxed{\;\tilde{\mathbf{x}}_k=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)\;}
$$

### 2.4 预测协方差 $\tilde{\mathbf{P}}_k$

记误差 $\delta\mathbf{x}_{k-1}=\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}$。由 $(\star)$ 与 $\tilde{\mathbf{x}}_k$ 的结果：

$$
\mathbf{x}_k-\tilde{\mathbf{x}}_k=\mathbf{F}\,\delta\mathbf{x}_{k-1}+\mathbf{w}_k
$$

按 **协方差矩阵的定义** $\tilde{\mathbf{P}}_k=E\!\left[(\mathbf{x}_k-\tilde{\mathbf{x}}_k)(\mathbf{x}_k-\tilde{\mathbf{x}}_k)^T\right]$ 展开：

$$
\tilde{\mathbf{P}}_k=
E\!\left[
\mathbf{F}\delta\mathbf{x}_{k-1}\delta\mathbf{x}_{k-1}^T\mathbf{F}^T
+\mathbf{F}\delta\mathbf{x}_{k-1}\mathbf{w}_k^T
+\mathbf{w}_k\delta\mathbf{x}_{k-1}^T\mathbf{F}^T
+\mathbf{w}_k\mathbf{w}_k^T
\right]
$$

由于 $\mathbf{w}_k$ 与 $\mathbf{x}_{k-1}$ 独立 $\Rightarrow E[\delta\mathbf{x}_{k-1}\mathbf{w}_k^T]=\mathbf{0}$，交叉项消失。再由：

$$
E[\delta\mathbf{x}_{k-1}\delta\mathbf{x}_{k-1}^T]=\hat{\mathbf{P}}_{k-1},\qquad
E[\mathbf{w}_k\mathbf{w}_k^T]=\mathbf{R}_k
$$

得到：

$$
\boxed{\;\tilde{\mathbf{P}}_k=\mathbf{F}\,\hat{\mathbf{P}}_{k-1}\,\mathbf{F}^T+\mathbf{R}_k\;}
$$

> **注（非加性噪声情形）**：若噪声以 $f(\mathbf{x}_{k-1},\mathbf{u}_k,\mathbf{w}_k)$ 方式进入模型（如题 3 中过程噪声乘以输入矩阵），则需引入噪声雅可比 $\mathbf{G}=\partial f/\partial\mathbf{w}$，结果变为 $\tilde{\mathbf{P}}_k=\mathbf{F}\hat{\mathbf{P}}_{k-1}\mathbf{F}^T+\mathbf{G}\mathbf{Q}\mathbf{G}^T$。教材式 (8.32) 通过将 $\mathbf{G}\mathbf{Q}\mathbf{G}^T$ 整体记为 $\mathbf{R}_k$，与上述加性形式一致。

---

## 题 3：EKF 雅可比矩阵 $\mathbf{F}$ 与 $\mathbf{H}$

状态、输入、地标：

$$
\mathbf{x}_k=\begin{bmatrix}x_k\\ y_k\\ \theta_k\end{bmatrix},\quad
\mathbf{u}_k=\begin{bmatrix}v_k\\ \omega_k\end{bmatrix},\quad
\mathbf{y}^l=\begin{bmatrix}x_l\\ y_l\end{bmatrix}
$$

### 3.1 运动模型 $\Rightarrow$ 雅可比 $\mathbf{F}$

题给运动模型（取 $\mathbf{w}_k=\mathbf{0}$ 用于求 $\mathbf{F}$）：

$$
\mathbf{x}_k=\mathbf{x}_{k-1}+\Delta t
\begin{bmatrix}\cos\theta_{k-1}&0\\ \sin\theta_{k-1}&0\\ 0&1\end{bmatrix}
\begin{bmatrix}v_k\\ \omega_k\end{bmatrix}
$$

逐分量展开：

$$
\begin{aligned}
x_k       &= x_{k-1} + \Delta t\,v_k\cos\theta_{k-1}\\
y_k       &= y_{k-1} + \Delta t\,v_k\sin\theta_{k-1}\\
\theta_k  &= \theta_{k-1} + \Delta t\,\omega_k
\end{aligned}
$$

#### 一阶近似形式

在 $(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)$ 处展开：

$$
f(\mathbf{x}_{k-1},\mathbf{u}_k)\approx
f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)+\mathbf{F}\,(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1})
$$

#### 求 $\mathbf{F}=\partial f/\partial\mathbf{x}_{k-1}$

逐项求偏导：

| | $\partial/\partial x_{k-1}$ | $\partial/\partial y_{k-1}$ | $\partial/\partial \theta_{k-1}$ |
|---|---|---|---|
| $x_k$ | $1$ | $0$ | $-\Delta t\,v_k\sin\theta_{k-1}$ |
| $y_k$ | $0$ | $1$ | $\;\;\Delta t\,v_k\cos\theta_{k-1}$ |
| $\theta_k$ | $0$ | $0$ | $1$ |

因此：

$$
\boxed{\;
\mathbf{F}=
\begin{bmatrix}
1 & 0 & -\Delta t\,v_k\sin\hat\theta_{k-1}\\[4pt]
0 & 1 & \;\;\;\Delta t\,v_k\cos\hat\theta_{k-1}\\[4pt]
0 & 0 & 1
\end{bmatrix}
\;}
$$

（在估计点 $\hat\theta_{k-1}$ 与里程计输入 $v_k$ 处求值。）

### 3.2 测量模型 $\Rightarrow$ 雅可比 $\mathbf{H}$

#### 中间变量

$$
\Delta x = x_l-x_k-d\cos\theta_k,\qquad
\Delta y = y_l-y_k-d\sin\theta_k
$$

$$
r=\sqrt{\Delta x^2+\Delta y^2},\qquad
r^2=\Delta x^2+\Delta y^2
$$

测量模型：

$$
h(\mathbf{x}_k,\mathbf{y}^l)=
\begin{bmatrix}
r\\[3pt]
\operatorname{atan2}(\Delta y,\Delta x)-\theta_k
\end{bmatrix}
$$

#### 一阶近似形式

在 $\tilde{\mathbf{x}}_k$ 处展开：

$$
h(\mathbf{x}_k,\mathbf{y}^l)\approx
h(\tilde{\mathbf{x}}_k,\mathbf{y}^l)+\mathbf{H}\,(\mathbf{x}_k-\tilde{\mathbf{x}}_k)
$$

#### 关键偏导数

辅助导数：

$$
\frac{\partial \Delta x}{\partial x_k}=-1,\quad
\frac{\partial \Delta y}{\partial y_k}=-1,\quad
\frac{\partial \Delta x}{\partial \theta_k}=d\sin\theta_k,\quad
\frac{\partial \Delta y}{\partial \theta_k}=-d\cos\theta_k
$$

##### (a) 距离 $r$ 部分

$$
\frac{\partial r}{\partial x_k}=\frac{1}{r}\!\left(\Delta x\cdot(-1)+\Delta y\cdot 0\right)=-\frac{\Delta x}{r}
$$

$$
\frac{\partial r}{\partial y_k}=-\frac{\Delta y}{r}
$$

$$
\frac{\partial r}{\partial \theta_k}
=\frac{1}{r}\!\left(\Delta x\cdot d\sin\theta_k+\Delta y\cdot(-d\cos\theta_k)\right)
=\frac{d}{r}(\Delta x\sin\theta_k-\Delta y\cos\theta_k)
$$

##### (b) 方位 $\phi=\operatorname{atan2}(\Delta y,\Delta x)-\theta_k$ 部分

利用

$$
\frac{\partial\operatorname{atan2}(v,u)}{\partial u}=-\frac{v}{u^2+v^2},\qquad
\frac{\partial\operatorname{atan2}(v,u)}{\partial v}=\frac{u}{u^2+v^2}
$$

链式法则：

$$
\frac{\partial\phi}{\partial x_k}
=-\frac{\Delta y}{r^2}\cdot(-1)+\frac{\Delta x}{r^2}\cdot 0
=\frac{\Delta y}{r^2}
$$

$$
\frac{\partial\phi}{\partial y_k}
=-\frac{\Delta y}{r^2}\cdot 0+\frac{\Delta x}{r^2}\cdot(-1)
=-\frac{\Delta x}{r^2}
$$

$$
\begin{aligned}
\frac{\partial\phi}{\partial\theta_k}
&=-\frac{\Delta y}{r^2}\cdot d\sin\theta_k+\frac{\Delta x}{r^2}\cdot(-d\cos\theta_k)-1\\[4pt]
&=-\frac{d}{r^2}\!\left(\Delta x\cos\theta_k+\Delta y\sin\theta_k\right)-1
\end{aligned}
$$

#### 最终 $\mathbf{H}$（$2\times 3$ 矩阵）

$$
\boxed{\;
\mathbf{H}=
\begin{bmatrix}
-\dfrac{\Delta x}{r} &
-\dfrac{\Delta y}{r} &
\;\;\dfrac{d}{r}\!\left(\Delta x\sin\theta_k-\Delta y\cos\theta_k\right)\\[14pt]
\;\;\dfrac{\Delta y}{r^2} &
-\dfrac{\Delta x}{r^2} &
-\dfrac{d}{r^2}\!\left(\Delta x\cos\theta_k+\Delta y\sin\theta_k\right)-1
\end{bmatrix}
\;}
$$

其中

$$
\Delta x=x_l-x_k-d\cos\theta_k,\quad
\Delta y=y_l-y_k-d\sin\theta_k,\quad
r=\sqrt{\Delta x^2+\Delta y^2}
$$

所有项在预测位姿 $\tilde{\mathbf{x}}_k=(\tilde x_k,\tilde y_k,\tilde\theta_k)$ 与已知地标 $\mathbf{y}^l$ 处求值。

---

## 答案速览

| 题号 | 关键结果 |
|---|---|
| 1 | $L(x)=(2\pi\sigma^2)^{-N/2}\exp\!\left[-\dfrac{1}{2\sigma^2}\sum_k(z_k-x)^2\right]$ |
| 1 | $\mathcal{L}(x)=\dfrac{N}{2}\ln(2\pi\sigma^2)+\dfrac{1}{2\sigma^2}\sum_k(z_k-x)^2$ |
| 1 | $x^{*}=\dfrac{1}{N}\sum_k z_k=\dfrac{14}{3}\approx 4.667$ |
| 2 | $\tilde{\mathbf{x}}_k=f(\hat{\mathbf{x}}_{k-1},\mathbf{u}_k)$ |
| 2 | $\tilde{\mathbf{P}}_k=\mathbf{F}\hat{\mathbf{P}}_{k-1}\mathbf{F}^T+\mathbf{R}_k$ |
| 3 | $\mathbf{F}=\begin{bmatrix}1&0&-\Delta t\,v_k\sin\hat\theta_{k-1}\\0&1&\Delta t\,v_k\cos\hat\theta_{k-1}\\0&0&1\end{bmatrix}$ |
| 3 | $\mathbf{H}=\begin{bmatrix}-\Delta x/r & -\Delta y/r & (d/r)(\Delta x\sin\theta_k-\Delta y\cos\theta_k)\\ \Delta y/r^2 & -\Delta x/r^2 & -(d/r^2)(\Delta x\cos\theta_k+\Delta y\sin\theta_k)-1\end{bmatrix}$ |

— 完 —
