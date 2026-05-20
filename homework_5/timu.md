**EE5346 - 自主机器人导航（2026春季）**  
**作业 #5（5%）**  
**截止日期：2026年6月1日（星期一）23:59**

**MLE 与扩展卡尔曼滤波器（MLE and Extended Kalman Filter）**

### 1.  
假设测量方程为 \( z_k = x + n \)，其中 \( z_k \) 为测量值，\( n \) 是服从 \( \mathcal{N}(0, \sigma) \) 的噪声，\( x \) 是待估计的机器人位置。  
现获得 \( N \) 个独立的机器人测量值 \( z_1, z_2, \dots, z_N \)。  
请推导机器人位置的最大似然估计（maximum likelihood estimate）。  
即：  
- 似然函数 \( L(x) = p(Z \mid x) \)（其中 \( Z \) 是 \( N \) 个测量值组成的向量）是什么？  
- 负对数似然 \( \mathcal{L}(x) = -\ln L(x) \) 又是什么？  

若 \( N=3 \)，且测量值为 \( z_1=3 \)，\( z_2=4 \)，\( z_3=7 \)，求机器人位置的 MLE \( x^* \)，其中 \( x^* = \arg\min \mathcal{L}(x) \)。  
**要求展示详细推导步骤。**

### 2.  
利用向量函数的均值向量（mean vector）和协方差矩阵（covariance matrix）的定义，推导教材第 202 页的式 (8.32)。  
即证明：在扩展卡尔曼滤波器（extended Kalman filter）中，预测状态的均值与协方差分别为：  
\[
\tilde{\mathbf{x}}_k = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k), \qquad \tilde{\mathbf{P}}_k = \mathbf{F} \hat{\mathbf{P}}_{k-1} \mathbf{F}^T + \mathbf{R}_k
\]

### 3.  
本题要求推导用于 2D 机器人轨迹定位的扩展卡尔曼滤波器（EKF）的雅可比矩阵（Jacobian matrices）。  
机器人位姿为 \( \mathbf{x}_k = [x_k \; y_k \; \theta_k]^T \)，使用运动模型和已知点地标（point landmark）的测量进行估计。  

车辆配备 LiDAR 传感器，可对环境中单个点地标 \( \mathbf{y}^l = [x_l, y_l]^T \) 返回距离和方位测量值 \( \mathbf{z}_k^l \)。地标的全局位置 \( \mathbf{y}^l \) 已知（即地图已知）。  

车辆运动模型以线速度和角速度里程计读数 \( \mathbf{u}_k = [v_k, \omega_k]^T \) 为输入，输出 2D 机器人位姿 \( \mathbf{x}_k \)，模型定义如下：  
\[
\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k, \mathbf{w}_k) = \mathbf{x}_{k-1} + \Delta t \begin{bmatrix}
\cos\theta_{k-1} & 0 \\
\sin\theta_{k-1} & 0 \\
0 & 1
\end{bmatrix}
(\mathbf{v}_k + \mathbf{w}_k), \qquad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q})
\]  
其中 \( \Delta t \) 为固定的采样时间间隔，过程噪声 \( \mathbf{w}_k \) 服从零均值、协方差为 \( \mathbf{Q} \) 的高斯分布。  

测量模型将当前车辆位姿与 LiDAR 的距离-方位测量 \( \mathbf{z}_k^l = h(\mathbf{x}_k, \mathbf{y}^l) = [r \; \phi]^T \) 关联（参考教材第 17 页式 (1.4)）：  
\[
\mathbf{z}_k^l = h(\mathbf{x}_k, \mathbf{y}^l) = \begin{bmatrix}
\sqrt{(x_l - x_k - d \cos\theta_k)^2 + (y_l - y_k - d \sin\theta_k)^2} \\[6pt]
\atantwo(y_l - y_k - d \sin\theta_k,\; x_l - x_k - d \cos\theta_k) - \theta_k
\end{bmatrix} + \mathbf{n}_k^l
\]  
其中 \( \mathbf{y}^l = [x_l, y_l]^T \) 为已知地标坐标，\( d \) 为机器人中心到激光测距仪的已知距离，测量噪声 \( \mathbf{n}_k^l \sim \mathcal{N}(0, \mathbf{R}) \)。  

**任务**：  
推导扩展卡尔曼滤波器定义的雅可比矩阵 \( \mathbf{F} \) 和 \( \mathbf{H} \)（即教材式 (8.28) 和 (8.30)）。  
**注意**：只需写出模型关于相关变量的一阶近似形式，以便直接计算雅可比矩阵 \( \mathbf{F} \) 和 \( \mathbf{H} \)。  

---

**作业要求**：  
- 请完整、清晰地展示所有推导步骤（包括似然函数、负对数似然、预测方程的均值与协方差推导，以及 \( \mathbf{F} \)、\( \mathbf{H} \) 的具体表达式）。  
- 所有数学表达式请使用标准符号和矩阵形式书写。  
- 提交时请附上详细计算过程。  

如需进一步澄清任何公式或需要我帮忙用中文详细讲解某一道题的解题思路，请随时告诉我！