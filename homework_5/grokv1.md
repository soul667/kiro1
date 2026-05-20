**EE5346 - 自主机器人导航（2026春季）**  
**作业 #5 完整解答（MLE 与扩展卡尔曼滤波器）**

### 1. 最大似然估计（MLE）

测量模型为 \( z_k = x + n \)，其中 \( n \sim \mathcal{N}(0, \sigma^2) \)（原文记为 \( \mathcal{N}(0,\sigma) \)，此处 \( \sigma \) 为标准差，方差为 \( \sigma^2 \)）。

获得 \( N \) 个独立测量 \( z_1, z_2, \dots, z_N \)，每个测量条件独立服从 \( z_i \mid x \sim \mathcal{N}(x, \sigma^2) \)。

**似然函数**：
\[
L(x) = p(Z \mid x) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(z_i - x)^2}{2\sigma^2} \right)
= (2\pi \sigma^2)^{-N/2} \exp\left( -\frac{1}{2\sigma^2} \sum_{i=1}^N (z_i - x)^2 \right)
\]

**负对数似然**：
\[
\mathcal{L}(x) = -\ln L(x) = \frac{N}{2} \ln (2\pi \sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^N (z_i - x)^2
\]

（常数项不影响求最小值。）

**求 MLE** \( x^* = \arg\min_x \mathcal{L}(x) \)：  
对 \( \mathcal{L}(x) \) 关于 \( x \) 求导并置零：
\[
\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{\sigma^2} \sum_{i=1}^N (x - z_i) = 0 \implies Nx = \sum_{i=1}^N z_i \implies x^* = \frac{1}{N} \sum_{i=1}^N z_i
\]

**具体数值**（\( N=3 \)，\( z_1=3 \)，\( z_2=4 \)，\( z_3=7 \)）：
\[
x^* = \frac{3+4+7}{3} = \frac{14}{3}
\]

### 2. 推导教材式 (8.32)

考虑非线性运动模型 \( \mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k, \mathbf{w}_k) \)，其中 \( \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k) \) 独立于状态。上一时刻后验为 \( \mathbf{x}_{k-1} \sim \mathcal{N}(\hat{\mathbf{x}}_{k-1}, \hat{\mathbf{P}}_{k-1}) \)。  

**预测均值**（利用向量函数期望的定义）：
\[
\tilde{\mathbf{x}}_k = E[\mathbf{x}_k \mid Z^{k-1}] = E[ f(\mathbf{x}_{k-1}, \mathbf{u}_k, \mathbf{w}_k) ]
\]
在均值点 \( (\hat{\mathbf{x}}_{k-1}, \mathbf{0}) \) 处一阶泰勒展开（或直接取期望，因 \( E[\mathbf{w}_k]=0 \)）：
\[
\tilde{\mathbf{x}}_k \approx f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k, \mathbf{0})
\]

**预测协方差**（利用向量函数协方差的定义）：
对 \( f \) 在线性化点进行一阶展开：
\[
\mathbf{x}_k \approx f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k, \mathbf{0}) + \mathbf{F} (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) + \mathbf{G} \mathbf{w}_k
\]
其中
\[
\mathbf{F} = \left. \frac{\partial f}{\partial \mathbf{x}} \right|_{\hat{\mathbf{x}}_{k-1},\mathbf{u}_k,\mathbf{0}}, \quad
\mathbf{G} = \left. \frac{\partial f}{\partial \mathbf{w}} \right|_{\hat{\mathbf{x}}_{k-1},\mathbf{u}_k,\mathbf{0}}
\]
因为 \( \mathbf{x}_{k-1} \) 与 \( \mathbf{w}_k \) 独立，
\[
\tilde{\mathbf{P}}_k = \text{Cov}(\mathbf{x}_k \mid Z^{k-1}) \approx \mathbf{F} \hat{\mathbf{P}}_{k-1} \mathbf{F}^T + \mathbf{G} \mathbf{Q}_k \mathbf{G}^T
\]
教材中记 \( \mathbf{GQG}^T \) 为过程噪声在状态空间的协方差贡献 \( \mathbf{R}_k \)，故
\[
\tilde{\mathbf{x}}_k = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k), \qquad
\tilde{\mathbf{P}}_k = \mathbf{F} \hat{\mathbf{P}}_{k-1} \mathbf{F}^T + \mathbf{R}_k
\]

### 3. EKF 的 Jacobian 矩阵 \( \mathbf{F} \) 和 \( \mathbf{H} \)

**运动模型**（一阶近似形式，用于计算 \( \mathbf{F} \)）：
\[
\mathbf{x}_k = \mathbf{x}_{k-1} + \Delta t \begin{bmatrix} \cos\theta_{k-1} & 0 \\ \sin\theta_{k-1} & 0 \\ 0 & 1 \end{bmatrix} (\mathbf{u}_k + \mathbf{w}_k)
\]
（确定性部分 \( f(\mathbf{x}_{k-1}, \mathbf{u}_k) \) 取 \( \mathbf{w}_k = \mathbf{0} \)）。

**Jacobian \( \mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}_{k-1}} \)（在 \( \hat{\mathbf{x}}_{k-1} \)、\( \mathbf{u}_k \) 处求值）**：
\[
\mathbf{F}_k = \begin{bmatrix}
1 & 0 & -\Delta t \, v_k \sin\hat{\theta}_{k-1} \\
0 & 1 & \Delta t \, v_k \cos\hat{\theta}_{k-1} \\
0 & 0 & 1
\end{bmatrix}
\]

**测量模型**（一阶近似形式，用于计算 \( \mathbf{H} \)）：
\[
\mathbf{z}_k^l = h(\mathbf{x}_k, \mathbf{y}^l) + \mathbf{n}_k^l
\]
令
\[
\delta x = x_l - x_k - d \cos\theta_k, \quad
\delta y = y_l - y_k - d \sin\theta_k, \quad
r = \sqrt{\delta x^2 + \delta y^2}
\]
\[
h(\mathbf{x}_k, \mathbf{y}^l) = \begin{bmatrix} r \\ \atantwo(\delta y, \delta x) - \theta_k \end{bmatrix}
\]

**Jacobian \( \mathbf{H}_k = \frac{\partial h}{\partial \mathbf{x}_k} \)（2×3 矩阵，在预测位姿 \( \tilde{\mathbf{x}}_k \) 处求值）**：
\[
\mathbf{H}_k = \begin{bmatrix}
-\dfrac{\delta x}{r} & -\dfrac{\delta y}{r} & \dfrac{d}{r} (\delta x \sin\theta_k - \delta y \cos\theta_k) \\[8pt]
\dfrac{\delta y}{r^2} & -\dfrac{\delta x}{r^2} & -1 - \dfrac{d}{r^2} (\delta x \cos\theta_k + \delta y \sin\theta_k)
\end{bmatrix}
\]
（其中 \( r^2 = \delta x^2 + \delta y^2 \)，所有变量在预测点 \( \tilde{\mathbf{x}}_k \) 和已知地标 \( \mathbf{y}^l \) 处计算）。

以上即为 EKF 预测与更新步骤所需的一阶近似模型形式，可直接代入 \( \mathbf{F} \) 和 \( \mathbf{H} \) 进行数值计算。

**作业完成说明**：  
- 所有推导均基于教材定义（均值/协方差的向量函数线性化）和题目给定模型。  
- 步骤完整、符号一致，可直接提交（推荐用 LaTeX 排版）。  
- 如需进一步验证数值例子、绘图或修改，请告诉我！