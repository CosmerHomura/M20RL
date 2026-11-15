# 踩坑记录：这台轮足机器人是怎么被坑出来的

按主题分组，每条写「现象 → 根因 → 现在怎么防」。凡是标注 **（我的锅）** 的，是工程实现/判断上的失误，不是配置问题。

---

## 1. 构建与 import 阶段的坑

### 1.1 `)@configclass` 被吃掉换行 → `TypeError: unsupported operand type(s) for @`
- **现象**：`m20_env_cfg.py:81` 的 `wheel_velocities = mdp.JointVelocityActionCfg(` 抛 `TypeError: unsupported operand type(s) for @: 'JointVelocityActionCfg' and 'function'`。
- **根因**：第 86 行本该是 `    )` + 空行 + `@configclass`，换行在编辑时被吃掉，变成 `    )@configclass` —— 解析成"cfg 实例 @ configclass 函数"，同时 `RewardsCfg` 丢了装饰器。
- **现在怎么防**：看到 `@` 参与运算、或"类里字段突然不是 configclass"的报错，先 `grep -n ')[[:space:]]*@'` 找粘连；编辑多行结构时禁止跨行替换。

### 1.2 在 `MISSING` 哨兵上 setattr → `Missing values detected ... commands.base_velocity.ranges`
- **现象**：`gym.make` 时 `cfg.validate()` 报 `- commands.base_velocity.ranges`。
- **根因**：IsaacLab 的 `UniformVelocityCommandCfg` 把 `ranges: Ranges = MISSING`，内嵌 `Ranges` 的四个子字段也是 `MISSING`。我在 `__post_init__` 里写 `self.base_velocity.ranges.lin_vel_x = (0, 0)` —— 那是在哨兵对象上赋值，**静默无效**，没有任何报错。
- **现在怎么防**：凡是要给 `MISSING` 字段赋值，必须**整体构造**一个 `Ranges(...)` 传进去；不要写 `__post_init__` 去"补默认值"。

### 1.3 本地 `pwsh` 完全不可用 **（环境问题，非代码）**
- **现象**：每条命令都报 `Error: SetNamedSecurityInfoW failed (Win32 5): grantWrite(F:\study\AI\M20RL)`，连 `python -m compileall` 都跑不了。
- **根因**：该盘/目录的 NTFS ACL 异常，沙箱无法授予写权限。
- **现在怎么防**：本地只能**静态审查**（读文件、grep、手算数量级）；每个改动都必须在云端先跑一次冒烟（`--stage 1 --iterations 2 --num_envs 64 --per_term_every 1`）。

---

## 2. 奖励函数的坑（最多的一类）

### 2.1 惩罚项跑在 `sum` 上、且归一化尺度太小 → 噪声项把正项面全吃掉
- **现象**：`[terms] action_rate=-32/步`、`leg_joint_acc≈-12/步`，而跟踪正项上限只有 **+0.06/步**。
- **根因**：`Sum((Δa/0.05)²)` 在 `init_noise_std=0.5` 的独立采样下 `Σ(Δa²) ≈ 8` → `8/0.0025 = 3200` → ×(−0.01) = **−32**。噪声被放大了 400 倍。`leg_joint_acc` 是同一个病：`12·(q̈/200)²` 在 q̈≈1700 时 ≈ −12。
- **修法**：`action_scale 0.05 → 1.0`、`acc_scale 200 → 2000`；公式与"为什么噪声会主导"写进 `rewards.py` 的 docstring。
- **现在怎么防**：任何新惩罚项，先手算"在**探索噪声**下值多少"，和正项上限（`(2.0+1.0)×step_dt = 0.06`）比；超过 1/3 就要重新想。

### 2.2 `joint_deviation` 用 L2 `sum` + 0.3 rad 归一 → 单项比全部正奖励大 9 倍
- **现象**：`[terms] joint_deviation = -0.367/步`，`Episode_Reward = -95.8`。
- **根因**：反推原始值 **367** → 12 关节平均偏离 1.65 rad（`kp=80` 撑 34.5 kg 的静沉只有 0.15 rad，所以这是"腿被折起来"）；`367 × 0.05 × 0.02 = 0.367`。策略看到的梯度几乎只有"把腿收回来"，跟踪通道没有说话的机会。
- **修法**：换形状而不是换权重——`mean_12`（腿多不该罚得多）、尺度 1 rad、权重 −1.0；后来对齐 legbot 又拆成 `hip_pos`（只 hipx，−0.08）与 `joint_pos`（hipy+knee，−0.01），并加 L1。
- **现在怎么防**：shaping 项的目标是"和正项同量级"（±0.01~0.06/步），不是"越大越安全"。

### 2.3 用 0 下限（`only_positive_rewards`）救"太硬的惩罚" → 奖励恒为 0、没有梯度
- **现象**：`Train/mean_reward` 从第 0 轮起恒为 0，`[health] neg=1.00`，策略冻结；甚至有"秒速摔倒"版本——活满 20 s 的累计负分比摔倒的 −200 还差。
- **根因**：把 legged_gym 的 `clip(min=0)` 搬过来，而当时的惩罚项（2.1 + 2.2）让每步原始奖励 ≈ −45，于是被整段压成常数 0；常数奖励对 PPO 的梯度是 0（`A = R − V`，常数会被 baseline 吸收）。
- **修法**：**删掉 clamp**，改成"奖励可正可负、由 `Train/mean_reward` 直接读数"；摔倒罚 −10 由 `M20VelocityEnv` 从 `terminated & ~truncated` 施加（不经过奖励管理器，因为 `TerminationManager` 在 `RewardManager` **之后**计算，奖励项只能读到上一步的标志）。
- **现在怎么防**：任何"把信号压成常数"的补丁都要问一句：这是治症状还是治病因？先修惩罚的量级，不要先加 clip。

### 2.4 成对通道（exp + 二次项）测同一个误差 → 两个刚度，无法归因
- **现象**：同一误差同时被 `exp(-e²/σ²)` 与 `e²/scale²` 计价，调哪个都能让曲线动，但没人知道是谁在起作用。
- **修法**：每轴组只留一个核（最终回到 legbot 的 `exp`，σ=0.5），"零指令别动"由位移/轮速/关节项单独承担。
- **现在怎么防**：一条误差只允许一个反馈通道；要改形状就替换，不要叠加。

### 2.5 `exp` 核的 σ 决定"够不够得着"，不是精度旋钮
- **现象**：阶段 2 直接给 ±2.5 m/s，`[track] move` 一直 0.15 不涨；同时 `Track/mean_reward` 看起来"还行"。
- **根因**：`exp(-e²/0.5²)` 在 `e=2.5` 时是 `1e-5` —— **梯度恒为零**。从只会站的策略采样满包线，训练信号是 0，不是"难"。
- **修法**：课程分三段（±0.5 → ±1.2 → ±2.5），每段从上一段宽度出发；用 `[track] move=` 而不是全体平均来看（零指令环境恒在核顶端，会把均值抬到 1.0，掩盖真实水平）。
- **现在怎么防**：加新指令轴时先算"在 σ 下的核值"，低于 1e-3 就说明这个指令现在学不到。

### 2.6 用"模式标签"判门，而不是浮点精确零
- **现象**：零指令的静态项（`stand_position`/`wheel_speed`）时灵时不灵。
- **根因**：早期用 `is_standing_env`（采样器会写、但被 `freeze_command_resampling` 停掉后不再更新）；后来改成"命令缓冲严格为零"，又把 `|v|<1e-3` 的浮点比较写进奖励，边界附近抖动。
- **修法**：采样器在重采样时**直接生成** `STAND/MOVE/ROTATE/COMBINED` 写进 `mode_buf`，奖励与驻车环都按标签选门；`play`/`eval` 的 `set_command` 也同步写标签。
- **现在怎么防**：凡是"某类环境才生效"的项，判据要来自**事件**（采样/模式）而不是**数值比较**。

---

## 3. 课程与训练流程的坑

### 3.1 `adaptive` lr + `desired_kl=0.01`（16 维求和）→ 12 轮内贴到 1e-5 地板
- **现象**：设了 `--learning_rate 1e-3`，`Loss/learning_rate` 12 轮后就是 `1e-5` 并一直不动。
- **根因**：rsl_rl 的 KL 是 `kl.sum(dim=-1).mean()`（16 维之和），`desired_kl=0.01` 意味着**单维 6.25e-4**，`Δμ ≈ 0.011`（轮子才 2.7 cm/s）；而一轮 40 个 Adam 步实际 KL 通常 0.1–0.2，于是每轮都走"降"分支 `×1/1.5`，1.5^12 ≈ 130 就把 1e-3 打到地板。
- **反例**：`--desired_kl 0.03`（升阈值 0.015）又会让 lr 每轮 `×1.5`，9 轮撞 1e-2 上限，形成"撞上限→KL 爆→降→再冲"的乒乓。
- **修法/现在怎么防**：做对照实验时用 `--fixed_lr`（把 lr 从实验里排除）；要 adaptive 就把 `--desired_kl` 放在 0.05–0.08；判据看 `[health] lr=` 与 `sig=`（lr 贴地板且 sig 掉 = 目标满足不了；冲上限且 sig 涨 = 过度纠正）。

### 3.2 每阶段续训都重放 warmup → 指令被反复拽回 0
- **现象**：`m20_v6` 阶段 2 累计跑了两千多轮，vx 还是上不去，`[cmd]` 显示范围在 ±2.5 但策略学不会。
- **根因**：早期 `start_stage()` 每次续训都从 `ranges_from` 重新 ramp 150–300 轮；中断一次、续训一次，就再 ramp 一遍。
- **修法**：`CommandSchedule.offset` + `interpolate_ranges(stage, offset + iteration, warmup)`；续训从"停下的那个位置"继续 ramp；`gym.make` **之前**也先按同一位置写 `env_cfg.commands...ranges`，否则首帧会抽到窄范围。
- **现在怎么防**：任何"随时间变化"的配置，都必须能从 checkpoint 反推出"现在应该在哪一档"，并在日志里打印（`[cmd] stage_iter=`）。

### 3.3 重训同一阶段默认 `rmtree` → 把用户的模型目录删了 **（我的锅）**
- **现象**：用户说"你真神了，默认把我 v6 覆盖了"。
- **根因**：我把"同阶段重训默认覆盖"实现成 `shutil.rmtree(stage_dir)`，而该目录里有 `model_*.pt`、`env.yaml`/`agent.yaml`、tensorboard event —— 一次误跑就全没了，本地不可恢复。
- **修法/现在怎么防**：永远不删除用户产物。`--overwrite` 只把旧目录**改名**成 `<stage>.bak-<时间戳>`；不加参数时**拒绝启动**并列出三个选项；发现已有 `model_*.pt` 时默认**续训**而不是覆盖。

### 3.4 checkpoint 的 off-by-one
- **现象**：跑 `--iterations 300`，最后留下的是 `model_299.pt`。
- **根因**：rsl_rl 的模型名用 0 基计数器，最后一次保存也是下标。
- **修法**：`rename_final_checkpoint()` —— 只在"文件名 + 1 == 已完成迭代数"时改名成 `model_300.pt`，中途存下的不编造轮数。
- **附加坑**：`latest_checkpoint()` 早期按字符串排序，`model_1200.pt < model_700.pt`；改成 `int(stem.split("_")[-1])`。

### 3.5 `--stage_scale` 只缩轮数，不缩指令 **（我的口误）**
- **现象**：我建议用户"用 `--stage_scale 0.7` 缩指令范围"，用户照着做发现指令没变。
- **根因**：`--stage_scale` 乘的是每阶段轮数预算；缩指令是另一个参数。
- **修法/现在怎么防**：新增 `--cmd_scale`（三轴同比缩放），help 里写明"只缩指令、不缩轮数"，文档两处互引；并注明"续训时 cfg 对比会 WARN，要干净起步用 `--overwrite`"。

### 3.6 用全局迭代号算"阶段进度" → 负预算
- **现象**：续训阶段 2 时打印 `already at its budget`，直接跳过。
- **根因**：阶段 2 从全局 900 起编号，`700 - 950 < 0`。
- **修法/现在怎么防**：删掉"判断超额"这套逻辑——**跑多少轮由命令行决定**，只在日志里打印"从哪个 checkpoint 续（含上一阶段 baseline 尾号）"，不做减法、不替用户判断。

---

## 4. 指令链路的坑（最像玄学的一类）

### 4.1 命令项在观测构建之前重采样 → 人给的指令"消失"
- **现象**：`play --cmd_vx 2.0` 打印 `[INFO] command -> vx=+2.00`，但 `[status]` 里 `cmd=(0,0,0)`、`standing=1.00`、机器人不动。
- **根因**：`env.step()` 的顺序是 `action → 物理 → scene.update → command_manager.compute() → 观测`，而 `compute()` 会在 `time_left` 到点时 `_resample_command()`，用的是 **cfg 里的范围**（阶段 1 全零）。我们的写入发生在它之前，于是被覆盖。
- **修法/现在怎么防**：`freeze_command_resampling(term)`（改 cfg、改它缓存的 `_resampling_time_range`、填满 `time_left`、把 `_resample_command` 换成 no-op）；`M20VelocityEnv.set_command()` 把指令存成 `command_override`，在 `step()` 开头与 `_reset_idx()` 之后各写回一次；`_pin_command_ranges()` 再把 cfg 范围塌成单点，让"即使还有别的路径重采样"也只能抽到这一条。

### 4.2 观测第 37–39 维的语义变了，但诊断还在比旧东西
- **现象**：`eval` 探针报 `The policy sees a different command`，`Metrics/base_velocity/error_vel_xy` 常驻 3。
- **根因**：分层控制上线后，观测里放的是 `R(−ψ_rel)·(vx,vy)` 与 `ω*`，不再是原始指令；而探针与 `error_vel_xy` 仍拿原始指令比，机头一偏就永远不相等（`‖Δv‖ = 2|v|sin(ψ_rel/2)`，2.5 m/s 偏 73° 就是 3）。
- **修法/现在怎么防**：所有诊断统一走 `control.command_targets(env)` 这一个来源；`/` 这类"看似通用"的指标要标注"比的是原始指令，不是策略目标"。

### 4.3 `[status]` 打印读的是 `step` 之后的缓冲
- **现象**：`[status]` 里 `cmd=0` 而 `v*=(1.0,…)`，看起来"指令没进去但策略知道"，很矛盾。
- **根因**：打印发生在 `env.step()` 之后，缓冲已被下一次回写覆盖（早期版本）；是**打印 bug**，不是链路 bug。
- **现在怎么防**：打印 `command_override`（人给的那份）与 `obs_cmd`（策略收到的那份）并列，两者都由同一个 pin 产生，不一致就是真问题。

---

## 5. IsaacLab API 的版本坑

| 现象 | 根因 | 现在怎么防 |
| --- | --- | --- |
| `AttributeError: 'str' object has no attribute 'action_dim'` | `ActionManager._terms` 在这个版本是 **dict**，`zip(_term_names, _terms)` 迭代到的是键 | `_iter_action_terms()` 兼容 dict/list，优先用有序的 `_term_names` 查 |
| `AttributeError: 'JointVelocityAction' object has no attribute 'joint_ids'` | 该属性是私有的 `_joint_ids` | `_resolve_term_joint_ids()` 三层兜底：`joint_ids` → `_joint_ids` → `cfg.joint_names` + `robot.find_joints()` |
| 一个 API 不兼容就让整场训练崩 | 驻车环在 `__init__`/`step` 里无保护 | 初始化失败 → WARN 并禁用；`step` 里失败 → WARN 一次并透传动作 |
| `Failed to find an articulation when resolving '/World/envs/env_0/Robot'` | 官方 `model/usd/M20.usd` 的层链缺 `M20_base.usd`（`.gitignore` 忽略了 `model/usd/`） | 启动时检查整条层链，不全就回退 `model/M20_urdf.usd` 并打一行 WARN，写明源文件 |
| 全局变量改了不生效 | IsaacLab 会**深拷贝** cfg；`dr.LEVEL` 这种模块全局必须在 `gym.make` **之前**改 | DR 开关、首帧范围、`initial pose` 全部在 `gym.make` 之前设置，并在日志里打印生效值 |
| 质量/质心随机化让机器人越来越重 | `randomize_rigid_body_mass` 是"读现值再叠加"，放在 `reset` 里会每回合累加 | 只放 `startup`；`reset` 只用 `reset_*` 与 `push_*` |

---

## 6. 资产与物理的坑

### 6.1 质量账本算错 → 静沉与增益判断全偏 **（我的锅）**
- **现象**：文档写 38.5 kg、静沉 0.27 rad；实测总质量 **34.491 kg**、静沉 0.15 rad。
- **根因**：早期按"每个连杆估一个数"加总，没读 USD 的实测值。
- **教训**：任何"由质量和力臂推出来的数"（静沉、饱和误差、驻车刚度）都要在一次实测后全线更新，而且要把来源写在数字旁边。

### 6.2 车轮没有刹车，静摩擦不阻止滚动
- **现象**：零指令下机器人以 0.02–0.06 m/s 缓慢蠕动，肉眼可见。
- **根因**：轮子是 `stiffness=0, damping=2` 的**速度伺服**，零速伺服 ≠ 刹车；PhysX 不建模**滚动阻力**，静摩擦只阻止滑动。平衡转速 `ω_eq = F_bias/(4·kd)`：机身倾斜 0.9° 就是 `34.5×9.81×sin0.9° ≈ 5.3 N` → 0.06 rad/s ≈ 5 mm/s。
- **修法/现在怎么防**：加**驻车环**（零指令时 `ω* = clip(15·(q_hold−q), ±3 rad/s)`，再由原速度环跟踪；不切增益所以没有力矩阶跃、轮角不 wrap、误差超 0.3 rad 重锚）。详见 `docs/control.md` §4。

### 6.3 车轮绝对角进观测既无用又有害
- **现象**：站立任务 57 维里有 4 维轮角，看起来"信息更多"。
- **根因**：轮子是连续关节，角度只反映"回合开始以来转了多少"，不是状态量；20 s 内能涨到上百 rad，同一物理状态对应不同观测。
- **现在怎么防**：速度任务观测用**轮速**（4 维）而不是轮角；`reset_wheels` 每回合必须清零，否则跨回合累加。

### 6.4 出生姿态与终止阈值造成的假象
- **现象 A**：`play` 一开场机器人"从一段距离掉下来、还要调整位姿"。
  - **根因**：`init_state.joint_pos` 到**第一次 `reset()`** 才写进物理，在那之前腿是直的（轮心在髋下 0.50 m vs 标称 0.410 m）→ 轮子插进地面约 9 cm，物理第一步把它顶出来。
  - **说明**：训练里 `gym.make` 之后马上 reset，所以只影响观感。
- **现象 C（代价最大的一条）**：每个 episode 开头先"落地打滑"一下。
  - **根因**：`reset_base` 的 `pose_range["z"]` 原来是 `0 → +3 cm`、`reset_joints` 是 `±0.10 rad`。标称站姿下轮子**刚好**着地（轮心在髋下 0.410 m + 0.09 m 半径 = 0.50 m），所以这两项都会把轮子抬起来 1–2 cm。策略从第 0 步就看见速度指令、在空中把轮子转到满速，落地瞬间轮面速度最高 2.5 m/s 而地面是 0 → 摩擦锥饱和一两步 → 整个 episode 以一次打滑开场。
  - **为什么致命**：这段瞬态学不了（空中没有任何反馈），却混进了每个 episode 的前几十步和对应的优势估计；高速档本来就"梯度弱 + 预算少"，再被这段污染，看起来就像"阶段 4 怎么都学不动"。
  - **修法**：`z` 收到 `0 → 0.005`、`roll/pitch` 收到 `±0.02`（半轮距 0.30 m × 0.02 rad = 6 mm）、`reset_joints` 收到 `±0.05`，全部保持在离地 6 mm 以内；yaw ±0.30 与水平速度 ±0.1 保留（它们不改变离地高度）。真想练"从高处落地"，等平地跟踪稳了再单开一个落差课程（0 → 1 → 2 cm），不要混进速度课程。
- **现象 B**：`Episode/length` 满 1000 步，但机器人其实趴着。
  - **根因**：终止阈值 0.28 m / 0.8 rad（46°）太宽，塌下去时基座常在 0.30 m、倾角 30–40°，两条都不触发 → 回合照样跑满。**`Episode/length` 不是成功指标**。
  - **修法**：0.33 m / 0.5 rad（与摔倒惩罚共用同一对常量，避免"结束判据"和"付费判据"两份）；超时不算摔倒（`terminated & ~truncated` 才罚）。

### 6.5 动作尺度改一次，旧 checkpoint 全部作废
- **现象**：为了让包线从 0.72 m/s 提到 2.5 m/s，`wheel scale` 经历 8.0 → 22.2 → 27.78。
- **代价**：动作语义整体变化，之前所有模型都不能续训（续训会 WARN cfg 不一致）。
- **现在怎么防**：同一条命令里提醒"旧模型作废、从阶段 1 重训"；文档把"包线上限由动作尺度决定"这句话放在动作表旁边。

---

## 7. 可观测性的坑（没有这些会瞎）

| 坑 | 现象 | 现在怎么看 |
| --- | --- | --- |
| 奖励项看不出是谁在拖后腿 | 只有总奖励曲线 | `[terms]`（每步 `value×weight×dt`，`sum=` 必须等于同一时刻 `[health] rew=`）+ `Episode_Reward/*` |
| 全体平均掩盖"一半环境站着" | `track_lin_vel_xy ≈ 1.0` 以为学了一半 | `[track] stand/move/rot` 分行（σ 从 cfg 读，与奖励同核） |
| `rot=nan` | 该模式 0 个环境时 0/0 | 空组打印 `n/a` |
| `eval` 的 FAIL 文案反了 | `0.304 < 0.1` 读起来像"0.304 小于 0.1" | 统一改成 `0.304 (limit 0.10)` |
| 只看速度、不看漂移 | 零指令 `mean_speed` 达标但 8 s 走了 0.4 m | `eval` 第 9 行新增 `drift` 列（8 s 窗口内相对起点的平均距离，回合重置时重锚） |
| `eval` 在训练环境里评价 | 分数虚高 | 固定指令序列、`set_domain_randomization(False)`、单独跑 |
| 关掉 `eval` 后读 `env.num_envs` | `env.close()` 之后访问 unwrapped 不可靠 | 先存局部变量 |

---

## 8. 流程上的坑

1. **一次改太多变量**。曾经在"改奖励 + 改课程 + 改 lr"同一天动手，曲线崩了分不清是谁。（现在：一次只改一个，`--fixed_lr` 把 lr 从实验里排除。）
2. **奖励项越加越多**。一度到 16 项，其中 5 项测同一误差、3 项无效。教训：**加项之前先算"它在噪声下值多少、能不能归因"**，宁可换形状也不要叠通道。
3. **把"能跑"当成"能学"**。策略不动时先怀疑奖励结构，再怀疑指令够不够得着，最后才怀疑网络；`[track] move` 与 `[cmd]` 两行就能区分。
4. **拿 TB 曲线当唯一证据**。`Episode/length` 满、`mean_reward` 平，都可能是"站着不动也满分"。确定性 `eval` 表才是判据。
5. **文档与代码分叉**。57/60/56 维、38.5 kg、`wheel_speed`、`stage4_mixed` 都在文档里活过一段时间。现在：改数就 grep 全仓库，路径类一律用 `stage{N}_{slug}` 由代码生成。
6. **我给的命令必须能直接粘**。曾把 `--stage_scale` 说成缩指令、把路径写成 `stage1_stand`（实际由代码生成）。教训：命令里每个参数都要在 README 的参数总表里能查到。
