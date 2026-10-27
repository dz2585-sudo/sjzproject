# AimRL 规格（Day 4）

## 研究问题与范围

在相同环境交互步数下，距离奖励塑形是否使 Q-learning 更快学会命中移动目标？训练完成后，目标移动概率变化如何影响表现？比较规则方法、基础奖励 Q-learning 和 potential-based 塑形 Q-learning。规则方法对准时开火，否则先调整横向再调整纵向。项目是小型可复现实验；塑形是已有方法，不声称发明新算法，也不预设强化学习优于规则方法。临时原型只表明方案可能可行，目前没有正式训练结果。

## 环境和状态

5×5 网格，x 向右、y 向下增大，坐标均为 0–4。每次 reset 随机选择不重合的准星 (xc,yc) 和目标 (xt,yt)；目标行 yt 在一次尝试中固定，初始方向随机向左或向右。状态 s=(xc,yc,xt,yt,v)，v=-1 为左、+1 为右；Q 表方向索引 0 为左、1 为右。Q 表 shape=(5,5,5,5,2,6)，dtype=float32，共 1,250 个状态组合、7,500 个 Q 值，元素数据 30,000 字节，不含 Python 进程其他开销。使用直接可见的坐标状态，观察空间与窗口像素无关。

| 编号 | 动作 | 效果 |
|---:|---|---|
| 0 | left | x 减 1 |
| 1 | right | x 加 1 |
| 2 | up | y 减 1 |
| 3 | down | y 加 1 |
| 4 | stay | 准星不动 |
| 5 | fire | 当前位置开火，准星不动 |

移动最多一格，越界时限制在边界。动作编号与配置 actions 顺序一致。

每步严格按以下顺序：

1. 准星执行动作。
2. 若 fire，用此刻准星与目标位置判断是否完全重合。
3. 命中立即终止，目标不再移动。
4. 未命中时，以 0.10 概率反转目标方向。
5. 再独立地以 0.25 概率尝试移动一格；若当前方向越界，先反向再移动一格。
6. 根据动作结果和目标更新后的真实下一状态计算奖励并返回下一状态。

命中为 terminated=True；50 步未命中为 truncated=True、terminated=False；第 50 步命中为 terminated=True、truncated=False。50 步是外部采样和评估上限，状态不包含剩余步数；超时评估记为未成功，不构造成有限时域终止状态。命中不 bootstrap；外部截断仍从截断前的下一状态 bootstrap，不使用 reset 后状态。未来接口为 reset 返回 (observation, info)，step 返回 (observation, reward, terminated, truncated, info)；本轮不引入 Gymnasium。

## 奖励与学习

基础奖励 rA：每步 -0.05，命中额外 +10（命中步 +9.95），空枪额外 -1（空枪步 -1.05），其他步 -0.05。

Phi(s) = -(|xc-xt| + |yc-yt|)；rB = rA + shaping_scale × [gamma × Phi(s_next) - Phi(s)]。shaping_scale=1.0，gamma 与 learning.gamma 共用。命中终止状态 Phi=0；时间截断不强制清零，下一状态采用目标更新后的真实状态。

Q(s,a) ← Q(s,a) + alpha × [target - Q(s,a)]。真正终止时 target=r；其他情况包括外部截断时 target=r+gamma×max_a Q(s_next,a)。两种奖励使用相同状态、动作、初值、训练预算和探索计划。训练时最大 Q 并列的动作随机选择；评估时按固定动作顺序打破并列。

## 实验协议（计划，尚未执行）

- 两种 Q-learning 各用根种子 0、1、2 训练，每次 100,000 环境交互步；alpha=0.15、gamma=0.95、初始 Q=0。epsilon 前 80,000 步从 1.00 线性下降至 0.05，此后保持。训练目标移动概率 0.25，方向反转概率 0.10。
- 从每个训练根种子构造互相独立的环境与探索随机数生成器。验证和测试使用独立场景种子，评估不能推进训练随机数状态。
- 第 0 步和每 10,000 步验证，200 个场景，种子 100000–100199，epsilon=0。
- 固定配置后，最终模型在移动概率 0.10、0.25、0.50 下各测试 500 次，场景种子 200000–200499。规则方法在每档速度上用同一组测试场景评估一次，不复制为三次独立训练。评估冻结 Q 表，不更新；最终测试数据不用于调参。
- 指标为 50 步内成功率、每次尝试平均步数（超时计 50）及平均空枪数。连续两个验证检查点成功率均 ≥90% 时，学习效率记录其中第一个检查点的训练步数；未达到记为 >100000。先分别汇总每个训练种子，再计算三个种子间均值和样本标准差。不同奖励版本不直接比较累计奖励。

## 交付边界与参考资料

课程最终交付计划为 ICML 双栏报告，正文最多 5 页（参考文献和成员贡献不计入），以及 GitHub 代码与复现说明。Day 4 已实现势函数塑形和单种子同预算开发对照；多种子正式实验和最终测试留待 Day 5。

实现接口为 `AimEnv(config)`、`reset(seed=None)` 和 `step(action)`。reset 的整数种子重置环境独立随机流；不传种子继续现有随机流。观察为不可变 tuple，info 包括本步的 step_count、hit、miss 和 base_reward。结束后需显式 reset。规则策略只读取观察，随机策略使用独立生成器；Q-learning 演示加载已训练模型。演示、测试和诊断命令见 README。

本次在 Windows 11、Python 3.13.9、NumPy 2.5.4、Matplotlib 3.11.2 上，配置检查和 10 个单元测试通过。各 100 次、种子 300000–300099 的诊断：规则策略成功率 100%、平均步数 5.07、平均空枪 0.00；随机策略对应 25%、43.77、6.89。CSV 与最终画面保存在 outputs；它们不是正式实验结果。交互窗口未验证。PNG 已检查；导出时用户目录的 Matplotlib 字体缓存无法写入，但图片生成成功。

Day 3 将训练环境与探索流由根种子派生，验证采用独立环境和固定 100000–100199 场景，不更新 Q 表或推进训练流。epsilon 按动作选择前已完成的全局训练步数计算；命中不 bootstrap，截断从该转移的下一状态 bootstrap。检查点、逐场景验证、基线与训练记录保存在 `outputs/day3/`。移动靶 seed 0 的 100,000 步单次开发试验验证成功率为 0%→100%，不是多种子正式结果；静止靶 10,000 步诊断为 0%→13.5%。项目内 `.mplconfig` 解决了本次无界面绘图缓存警告；交互窗口仍未验证。运行命令与模型配套加载规则见 README。

Day 4 环境仍只返回基础奖励，训练按配置仅变换一次学习奖励；日志三种回报为未折扣尝试内求和。两个 seed=0 开发运行各 100,000 步，使用相同验证种子，基础版复现 Day 3。第 10,000 步基础/塑形成功率为 70%/100%，平均步数 26.605/6.990；连续两个检查点至少 90% 的第一个检查点分别为 20,000/10,000 步（分辨率 10,000）。仅为单种子初步观察，不代表统计显著。最终测试未使用；Day 5 冻结协议见 `outputs/day4/protocol_frozen.json`。静止靶重放说明见 `outputs/day4/static_diagnostic_note.md`。

- Ng, Harada & Russell (1999), *Policy invariance under reward transformations: Theory and application to reward shaping*: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- Gymnasium, *Handling Time Limits*: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- Gymnasium, *Solving Frozenlake with Tabular Q-Learning*: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
