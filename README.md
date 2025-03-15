# Deep Constrained Reinforcement Learning (WIP)

---

基于原始对偶（primal-dual）方法的约束强化学习（Constrained Reinforcement Learning, CRL）的数学模型，主要通过约束马尔可夫决策过程（CMDP）​框架和拉格朗日松弛技术将带约束的优化问题转化为无约束的优化目标，并交替更新策略参数（原始变量）和对偶变量。以下是其核心数学模型及步骤：

---

**1. 约束马尔可夫决策过程（CMDP）的数学定义**

定义智能体遵循策略$\pi$所获得的长期奖励为
$$
J_R(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right],
$$
其中$r(s_t,a_t)$为奖励函数，$\gamma \in(0，1]$是折扣因子，$\tau=(s_0,a_0,s_1,a_1,\ldots)$为样本轨迹。定义第$i$个约束$c_i(s_t, a_t)$的长期代价为
$$
J_{C_i}(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t c_i(s_t, a_t)\right]  \quad (i=1,2,\ldots,m)
$$
CMDP在标准MDP基础上增加了长期代价约束，目标是在满足所有$m$个代价约束条件下最大化长期奖励。其数学形式为：
$$
\begin{equation}
\begin{aligned}
\max_{\pi} \quad & J_R(\pi)\\
\textrm{s.t.} \quad & J_{C_i}(\pi) \leq d_i \quad (i=1,2,\ldots,m),
\end{aligned}
\end{equation}
$$
其中$d_i$是第$i$个约束对应的约束阈值。

---

**2. 拉格朗日松弛与对偶问题**

通过引入拉格朗日乘子$\pmb{\lambda} =(λ_1, \lambda_2,\ldots,\lambda_i,\ldots,\lambda_m)^\top \in \mathbb{R}^m_{+}$，将原约束问题转化为无约束的min-max优化问题。松弛后的拉格朗日目标函数为：
$$
\mathcal{L}(\pi,\lambda) \triangleq J_R(\pi) - \pmb{\lambda}^\top (J_{\pmb{C}}(\pi)-\pmb{d}),
$$
其中$J_{\pmb{C}}=(J_{C_1},J_{C_2},\ldots,J_{C_m})^\top$，$\pmb{d}=(d_1,d_2,\ldots,d_m)^\top$。原问题为：
$$
\max_{\pi}\min_{\pmb{\lambda}\in \mathbb{R}^m_{+}} \mathcal{L}(\pi,\lambda),
$$
对偶问题为:
$$
\min_{\pmb{\lambda}\in \mathbb{R}^m_{+}} \max_{\pi} \mathcal{L}(\pi,\lambda),
$$
该形式将原问题分解为原始变量（策略$\pi$）和对偶变量（乘子$\pmb{\lambda}$）的交替优化。

> 文献中通常有两种表达约束的方法：代价函数(cost)和效用函数(utility)。代价一般是指小于某个阈值的约束形式，如$c_i(s, a) \leq d_i$，而效用（获益）则是指大于某个阈值的约束形式，即$g_i(s, a) \geq d_i$。对于效用函数来说，需要相应的改变拉格朗日目标函数中乘子的符号，即
> $$
\mathcal{L}(\pi,\lambda) \triangleq J_R(\pi) + \pmb{\lambda}^\top (J_{\pmb{G}}(\pi)-\pmb{d}),
$$

> Soft Actor-Critic (SAC) 等价于解决如下CRL问题：
$$
\begin{equation}
\begin{aligned}
\max_{\pi} \quad & J_R(\pi)\\
\textrm{s.t.} \quad & \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \bigg[ -\log\big(\pi_t(a_t|s_t)\big) \bigg] \geq \mathcal{H} \quad \forall t,
\end{aligned}
\end{equation}
$$
> 其中$\mathcal{H}$为所期望达到的策略分布信息熵的阈值，$\rho_\pi(s, a)$表示由策略$\pi(\cdot|s)$引起的轨迹分布的状态和状态动作的边际分布(marginal)，此时随机策略$\pi$的信息熵为约束对应的效用函数。

---

**3. 理论保证：零对偶间隙**

在满足*Slater条件*（存在至少一个严格可行策略）时，原问题与对偶问题具有零对偶间隙，即：

$$
\max_{\pi}\min_{\pmb{\lambda}} \mathcal{L}(\pi,\lambda) = \min_{\pmb{\lambda}} \max_{\pi} \mathcal{L}(\pi,\lambda),
$$
这一性质保证了对偶问题的最优解与原问题一致，即使原问题非凸。

---

**4. Q函数、V函数与贝尔曼方程**

在CRL中，奖励和约束相关的$Q$函数和价值函数的定义与标准RL类似，其中奖励和第$i$个约束对应的$Q$函数用于衡量策略在状态$s$下采取动作$a$的长期（奖励/代价）期望：
$$
\begin{align}
Q^{\pi}_\diamond(s,a)&=\mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t \cdot \diamond(s_t,a_t) \bigg\vert s_t=s, a_t=a \right],
\end{align}
$$
其中$\diamond$ 为占位符，代指相应的奖励函数或者代价函数。类似的，价值函数用于衡量策略在状态$s$下的长期（奖励/代价）期望：
$$
\begin{align}
V^{\pi}_\diamond(s)&=\mathbb{E}_{\pi}\left[ \sum_{t=0}^{\infty} \gamma^t \cdot \diamond(s_t,a_t) \bigg\vert s_t=s \right]=\mathbb{E}_{a \sim \pi(\cdot|s)} \left[ Q^{\pi}_\diamond(s,a) \right],
\end{align}
$$
贝尔曼方程分别为
$$
\begin{align}
Q^{\pi}_\diamond(s,a)&=\diamond(s, a) + \gamma \mathbb{E}_{s^{\prime} \sim P(\cdot|s, a)}\left[ V^{\pi}_\diamond(s^{\prime}) \right],
\end{align}
$$
其中$s^{\prime} \sim P(\cdot|s, a)$为状态转移函数。

---

**5. 相关论文与代码实现**

在Actor-Critic框架代码实现中，通常通过以下方式隐式处理V函数：

- Critic网络直接建模Q函数，输出 $Q_{\diamond}(s,a)$，无需显式计算 $V_{\diamond}(s)$。
- 在Actor更新时，通过动作采样隐式估计V函数
  - 策略$\pi$采样动作 $a \sim \pi(\cdot∣s)$，进而计算$Q^{\pi}_\diamond(s,a)$；
  - 状态价值$V_\diamond(s)$通过$\mathbb{E}_{a \sim \pi(\cdot|s)} \left[ Q^{\pi}_\diamond(s,a) \right]$近似。

现有研究遵循上述Primal-Dual框架，进行了一系列的改进，相关论文和具体实现代码参考：

- [1] The Fast Safe Reinforcement Learning (FSRL) package (包含了许多基于Primal-Dual的主流算法实现)
  - 代码: https://github.com/liuzuxin/fsrl
- [2] Off-Policy Primal-Dual Safe Reinforcement Learning （ICLR, 2024)
  - 论文: https://arxiv.org/pdf/2401.14758
  - 代码: https://github.com/ZifanWu/CAL/tree/main 
- [3] Safe Policies for Reinforcement Learning via Primal-Dual Methods (IEEE Transactions on Automatic Control, 2023)
  - 论文：https://arxiv.org/pdf/1911.09101
- [4] Provably Efficient Safe Exploration via Primal-Dual Policy Optimization (AISTATS, 2021, 理论性研究)
  - 论文：https://arxiv.org/pdf/1911.09101
