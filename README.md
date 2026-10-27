# 6885finalproject_dtz

# AimRL

AimRL 是 ELEN E6885 课程的小型可复现强化学习项目，研究距离奖励塑形对命中移动目标的 Q-learning 学习效率的影响，以及目标移动概率变化对表现的影响。比较规则方法、基础奖励和塑形奖励；不预设哪种方法更好。

目前已完成 Day 5：网格环境、基础与塑形 Q-learning、三种子实验、独立最终测试及结果汇总。无需 GPU、CUDA、PyTorch 或 TensorFlow。

## 软件环境与安装

需要 Python 3 和项目专用虚拟环境。核心依赖是 NumPy 和 Matplotlib。已在本次执行环境的 Windows、Python 3.13.9 虚拟环境中验证 NumPy 2.5.4 和 Matplotlib 3.11.2，并将这两个版本固定在 requirements.txt；其他平台和 Python 版本尚未验证。

在项目根目录运行：

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_setup.py
```

macOS / Linux：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/check_setup.py
```

不需要激活虚拟环境或修改 PowerShell 执行策略。检查脚本打印操作系统、解释器、实际导入的依赖版本、配置和资源预算；依赖或配置出错时返回非零状态。它不训练或打开窗口。

## 环境与演示

`AimEnv(config)` 的 `reset(seed=42)` 返回 `(observation, info)`，`step(action)` 返回 `(observation, reward, terminated, truncated, info)`。观察为 `(xc, yc, xt, yt, v)`，动作 0–5 依次为 left、right、up、down、stay、fire。每步先执行准星动作并判断射击，命中立即终止；未命中先抽样目标转向，再独立抽样移动。基础奖励为普通步 -0.05、空枪 -1.05、命中 +9.95。未命中达到上限为截断；上限步命中优先终止。结束后必须 reset。

在项目根目录的 Windows PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m aimrl.demo --policy rule --episodes 3 --seed 42 --interval-ms 250
.\.venv\Scripts\python.exe -m aimrl.demo --policy rule --episodes 100 --seed 300000 --headless --csv outputs/day2_rule_smoke.csv --save-frame outputs/day2_rule_preview.png
.\.venv\Scripts\python.exe -m aimrl.demo --policy random --episodes 100 --seed 300000 --headless --csv outputs/day2_random_smoke.csv
```

图形演示自动进行，结束后关闭窗口退出；`--policy random` 可换成随机策略。无界面模式不等待，也不创建 GUI；场景种子从指定 seed 逐次加一，随机策略使用独立随机流。CSV 和预览图是 Day 2 诊断，不是学习曲线或正式测试结果。

Day 2 的无界面诊断记录保留在 outputs；交互图形窗口尚未验证。

## Day 3 训练与验证

`QLearningAgent` 的 `state_index` 映射方向 -1→0、+1→1；`select_action` 在训练时随机打破最大值并列，`greedy_action` 以固定动作顺序打破并列。`update(state, action, reward, next_state, terminated, truncated)` 仅在命中终止时停止 bootstrap，截断仍使用真实下一状态。epsilon 以全局交互步数从 1 线性下降到 0.05；环境和探索使用 `SeedSequence(seed).spawn(2)` 派生的独立流。验证使用独立环境、固定种子和冻结 Q 表。

以下命令在项目根目录运行；训练拒绝覆盖非空输出目录，重复运行需指定新目录：

```powershell
.\.venv\Scripts\python.exe -m aimrl.train --config outputs/day3/static_diagnostic_config.json --seed 42 --reward-mode base --output outputs/day3/static_new
.\.venv\Scripts\python.exe -m aimrl.train --config configs/base.json --seed 0 --reward-mode base --output outputs/day3/base_new
.\.venv\Scripts\python.exe -m aimrl.plots --run-dir outputs/day3/base_seed0
.\.venv\Scripts\python.exe -m aimrl.demo --policy qlearning --model outputs/day3/base_seed0/q_table.npy --episodes 3 --seed 42 --interval-ms 250
.\.venv\Scripts\python.exe -m aimrl.demo --policy qlearning --model outputs/day3/base_seed0/q_table.npy --episodes 3 --seed 42 --headless --save-frame outputs/day3/base_seed0/qlearning_preview.png
```

演示读取最终 Q 表旁的 `run_config.json` 和 `metadata.json`，校验网格、动作和表的形状、dtype、数值，并使用保存的运行环境参数。图表和演示在导入 Matplotlib 前优先使用用户已设置且可写的 `MPLCONFIGDIR`，否则使用项目 `.mplconfig` 或临时目录，不修改系统环境变量。项目缓存已加入忽略列表；本次无界面导图未再出现字体缓存警告，GUI 尚未验证。

本次 Windows 11、Python 3.13.9、NumPy 2.5.4、Matplotlib 3.11.2 的单种子开发运行：静止靶 seed 42、10,000 步验证成功率从 0% 到 13.5%；移动靶 seed 0、100,000 步从 0% 到 100%。主运行记录实际 100,000 次交互，保存 11 个检查点及 2,200 行验证场景；模型保存后逐项和贪心动作重载核对通过。`episodes.csv` 仅含完整尝试，末尾 5 步未完成尝试记录在 metadata。结果不是多种子正式结论，最终测试集未使用。

运行目录内的 `run_config.json`、`metadata.json`、`q_table.npy`、`checkpoints/`、`episodes.csv`、`validation.csv`、`validation_episodes.csv`、`baselines.csv` 和 `learning_curve.png` 分别保存配置、元信息、模型、检查点、完整训练尝试、验证汇总与逐场景记录、同验证集基线和曲线。

## 文件职责

### Day 4 塑形与单种子开发对照

环境始终返回基础奖励。训练使用 `Phi(s) = -(|xc-xt| + |yc-yt|)`，potential 模式的更新奖励为基础奖励加 `scale × (gamma × Phi(next_state) - Phi(state))`；命中时下一势值取 0，时间截断保留真实下一势值并 bootstrap。两个模式均从全零 Q 表开始，验证冻结 Q 表且使用基础奖励。`episodes.csv` 中 base_return、learning_return、shaping_return 是各尝试内未折扣的奖励和；不以不同模式的 learning_return 大小衡量性能。

在项目根目录运行（训练目录非空时需指定新的输出路径）：

```powershell
.\.venv\Scripts\python.exe -m aimrl.train --config configs/base.json --seed 0 --reward-mode base --output outputs/day4/base_new
.\.venv\Scripts\python.exe -m aimrl.train --config configs/base.json --seed 0 --reward-mode potential --output outputs/day4/potential_new
.\.venv\Scripts\python.exe -m aimrl.demo --policy qlearning --model outputs/day4/potential_seed0/q_table.npy --episodes 3 --seed 42 --interval-ms 250
```

现有结果 `outputs/day4/comparison/` 来源于同配置、同 seed、各 100,000 步、200 个相同验证场景的单种子开发对照。第 10,000 步 base/potential 成功率为 70%/100%，平均步数 26.605/6.990；第 100,000 步两者成功率均为 100%，平均步数 5.055/4.670。连续两次验证成功率至少 90% 的首个检查点分别为 20,000/10,000 步，分辨率 10,000 步。这不是跨种子统计或显著性结论。Day 4 base 与 Day 3 的配置、最终 Q 表和各验证指标一致。

静止靶的只读重放及其局限记录在 `outputs/day4/static_diagnostic_note.md`。正式实验设置固定在 `outputs/day4/protocol_frozen.json`；最终测试集和多种子正式实验尚未使用。已重载塑形模型并完成无界面演示，交互窗口仍未验证。

- `spec.md`：环境、奖励、算法、实验协议、指标和参考资料。
- `configs/base.json`：后续实现使用的集中参数。
- `requirements.txt`：核心依赖。
- `scripts/check_setup.py`：独立于当前工作目录的配置和环境检查。
- `aimrl/__init__.py`：Python 包说明。
- `aimrl/config.py`、`aimrl/env.py`：读取配置和实现环境。
- `aimrl/policies.py`：规则与随机策略。
- `aimrl/render.py`、`aimrl/demo.py`：绘图与自动演示。
- `aimrl/agent.py`、`aimrl/train.py`、`aimrl/evaluate.py`：Q 表、训练与独立验证。
- `aimrl/rewards.py`、`aimrl/compare.py`：奖励变换和开发对照。
- `aimrl/plots.py`、`aimrl/mpl_setup.py`：验证曲线与可写绘图缓存。
- `tests/test_env.py`：环境和策略的标准库单元测试。
- `tests/test_learning.py`、`tests/test_rewards.py`：学习逻辑、训练流程与塑形测试。
- `outputs/`：本次真实诊断 CSV 与最终状态预览图。
- `.gitignore`：忽略虚拟环境、缓存和临时文件，保留正式数据。

## Day 5 正式实验与复现

正式运行保存于 `outputs/day5/`：两种方法各以种子 0、1、2 从零训练 100,000 步，随后固定最终模型，在独立场景 200000–200499 和移动概率 0.10、0.25、0.50 下测试。规则与随机基线分别按相同场景评估，不计入训练种子标准差。成功率、平均步数、平均空枪与基础回报使用全部回合；学习曲线和测试汇总的误差为三个训练种子的样本标准差。

结果、逐种子阈值、配对差值和局限见 `outputs/day5/summary/results_summary.md`；原始记录见 `outputs/day5/test/`，六个模型和验证记录见 `outputs/day5/runs/`。`protocol_used.json` 记录最终模型哈希、测试前清单与测试状态。seed=0 曾用于开发，测试场景种子独立，但不保证初始状态未在训练时出现。GUI 未验证。

在项目根目录运行以下命令以创建新的完整实验；训练和测试拒绝覆盖已有原始结果，汇总只读取已保存的记录：

```powershell
.\.venv\Scripts\python.exe scripts\check_setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m aimrl.experiments --protocol outputs/day4/protocol_frozen.json --output outputs/day5_new --stage train
.\.venv\Scripts\python.exe -m aimrl.experiments --protocol outputs/day4/protocol_frozen.json --output outputs/day5_new --stage test
.\.venv\Scripts\python.exe -m aimrl.experiments --protocol outputs/day4/protocol_frozen.json --output outputs/day5_new --stage summarize
```

不要用最终测试来调整模型、超参数或挑选检查点；三训练种子只支持描述性结论。


GitHub 归档说明：本仓库保留 Day 5 正式结果和复现所需的 Day 4 协议、回归模型及开发记录。历史 JSON 中的绝对路径保留为原始运行证据；在其他电脑复现时请使用上述命令生成新的输出目录。当前尚无完成的 report/ 报告。
