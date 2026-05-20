# Grok 的原始回答

## 1. 神经网络（线性激活函数）（30分）

### (1) 单个神经元（10分）

**平方误差函数：**

神经元输出为
$$ \hat{y} = g(\mathbf{x}) = c + b \cdot \frac{1}{n} \sum_{i=0}^{n} W_i x_i $$
（索引按题目 $x_0 \sim x_n$，形式一致）。

平方误差函数定义为：
$$ E(\mathbf{W}) = \frac{1}{2} (y - \hat{y})^2 $$

**权重更新规则（梯度下降）：**

对任意权重 $W_k$（$k = 0,1,\dots,n$）：

$$ \frac{\partial E}{\partial W_k} = -(y - \hat{y}) \cdot \left( b \cdot \frac{x_k}{n} \right) $$

梯度下降更新规则：
$$ W_k \leftarrow W_k + \eta \cdot (y - \hat{y}) \cdot \left( b \cdot \frac{1}{n} \right) x_k $$

其中 $\eta > 0$ 为学习率。

### (2) 具有一个隐藏层的网络等价于单层线性网络（20分）

**输出单元的表达式：**

隐藏层第 $j$ 个神经元（入度为 $n$）：
$$ h_j = c + b \cdot \frac{1}{n} \sum_{k=1}^{n} w_{k,j} \, x_k \quad (j=1,\dots,m) $$

输出神经元（入度为 $m$）：
$$ o = c' + b' \cdot \frac{1}{m} \sum_{j=1}^{m} W_j \, h_j $$

代入展开：

$$ o = c' + b'c + \frac{b'b}{mn} \sum_{k=1}^{n} \left( \sum_{j=1}^{m} W_j w_{k,j} \right) x_k $$

令常数 $C = c' + b'c$，缩放系数 $\beta = \frac{b'b}{mn}$，复合权重 $W'_k = \sum_{j=1}^{m} W_j w_{k,j}$，则：

$$ o = C + \beta \sum_{k=1}^{n} W'_k \, x_k $$

这与单个线性神经元的输出形式完全相同（只是偏置常数和缩放系数不同），因此可被无隐藏层的单层线性网络精确表示。

---

## 2. 马尔可夫决策过程（MDP）（30分）

### (1) 总奖励 $V$（8分）

$$ V = \sum_{t=1}^{\infty} \gamma^{t-1} R(s_t, a_t, s_{t+1}) $$

### (2) 值函数 $V^\pi(s_0)$（8分）

$$ V^\pi(s_0) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 \right] $$

### (3) 值迭代方法（14分）

1. 初始化 $V_0(s)$。
2. 迭代：$V_{k+1}(s) = \max_{a} \sum_{s'} T(s,a,s')[R(s,a,s') + \gamma V_k(s')]$
3. 直到 $\|V_{k+1}-V_k\|_\infty < \epsilon$ 收敛。
4. 提取最优策略：$\pi^*(s) = \arg\max_a \sum_{s'} T(s,a,s')[R(s,a,s') + \gamma V^*(s')]$

---

## 3. 启发式与搜索（40分）

### (1) 可采纳启发式组合（25分）

- (a) $h_1 + h_2$：不可采纳（可超过 $h^*$）
- (b) $h_1 \times h_2$：不可采纳
- (c) $\max(h_1,h_2)$：可采纳
- (d) $\min(h_1,h_2)$：可采纳
- (e) $\alpha h_1 + (1-\alpha) h_2$：可采纳

**结论**：保证可采纳的是 (c)、(d)、(e)。

### (2) 判断题（15分）

**判断：正确（True）。**

加正常数后，每条路径的额外代价 $= c \times$（边数），步数不同则增加量不同，故原最优路径可能不再最优。

**反例**：
- A：旧 10（3 步）→ $10 + 6 = 16$
- B：旧 12（1 步）→ $12 + 2 = 14$

加 $c = 2$ 后 B 成为最优。
