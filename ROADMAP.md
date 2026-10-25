# ROTA 开发路线图

**ROTA**：**R**otational **O**rder **T**racking & **A**nalysis

拉丁语 *rota* 意为"轮子"，与 rotate、rotation、rotor 同源。

ROTA 是一个轻量、开源、经过验证的 Python 阶次分析库，用于旋转机械（发动机、电机、齿轮箱等）的噪声与振动信号分析，并提供基于阶次的声音重合成工具。它只依赖 NumPy 和 SciPy。每个函数都依据公开文献独立实现，并附带验证脚本。

命名约定：正文中写作 ROTA；代码中的包名和导入名为小写 `rota`（`pip install rota`，`import rota`）。

许可证：待定（GPL-3.0 或 BSD-3-Clause），v0.1 发布前确定。

**本版修订（2026-10-25）**：在初稿基础上吸收了作者既往研究项目中的经验教训（第 6 节）。ROTA 不复用任何既往代码，全部函数按公开文献从零实现。主要变化：

- 新增 `rota.resynth`（重合成）、`rota.metrics`（验证指标）、`rota.tonal`（心理声学桥接）三个模块；
- `rota.vkf` 扩展为 `rota.extract`，在 Vold-Kalman 之外增加相位锁定的短时谐波提取器；
- `rota.speed` 增加无转速计的梳齿–Viterbi 转速估计、转速→相位、车速→转速、控制信号整理；
- 验证用例从 T1–T7 扩展到 T1–T14，版本计划改为 v0.1–v0.4 + v1.0；
- 独立实现规范从 MATLAB 扩展到所有商业软件、专有文件格式、许可不明的第三方代码和受限数据。

---

## 1. 设计原则

1. **独立实现**：算法只依据公开文献，不参照任何商业软件的源代码或内部数据结构（见第 4 节）。
2. **可验证**：每个函数都在已知解析真值的合成信号上测试。有商业软件可对照的部分，另做黑盒对照，并公开结果。
3. **透明**：docstring 写明所依据的文献、幅值定义（RMS 或峰值）和单位约定。估计类函数（转速、阶次幅值）返回置信度，并报告是否触碰了搜索边界。
4. **Python 风格**：函数用 snake_case 命名，输入输出为 NumPy 数组，计算与绘图分离。
5. **可衔接**：结果能直接送入 SQAT、MoSQITo、ECMA-418-2 开源实现等心理声学工具。
6. **多通道优先**：所有算子对每个通道独立、线性地作用，输入形状为 `(n,)` 或 `(n, ch)`，不改变通道间的幅相关系。Ambisonics、传声器阵列等多通道录音可以直接处理，提取或去除阶次后空间信息保持不变。
7. **可复现**：所有随机过程接受 `rng`（`numpy.random.Generator`）或整数种子；重合成输出可附带 JSON 侧车文件（参数、种子、版本、输入摘要）。
8. **无隐含常数**：增益、校准因子、参考声压一律显式传入，不在函数内部埋默认倍数。

---

## 2. 功能清单

模块划分：`rota.speed`、`rota.angle`、`rota.maps`、`rota.slices`、`rota.extract`、`rota.resynth`、`rota.synth`、`rota.metrics`、`rota.tonal`、`rota.plot`。

### 2.1 转速参考（`rota.speed`）

| 函数 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `tacho_to_rpm` | 转速计脉冲 → 转速曲线。支持每转多脉冲、亚采样边沿插值、丢脉冲与抖动处理、平滑 | Brandt 2011 | v0.1 |
| `rpm_to_phase` | 转速曲线 → 累计转角（圈数）。梯形积分；可选转速计延迟补偿；最小转速下限（避免 f→0 的奇异） | — | v0.1 |
| `regularize_control` | 转速、车速等控制信号整理：去毛刺（滑动中值 + 容差）、重采样到等间隔网格（如 100 Hz）、缺口置 NaN、与音频时间轴对齐；支持与声学通道成整数比的慢速通道（如 5:1） | — | v0.1 |
| `speed_to_rpm` | 车速 → 转速。两种模型：集总斜率（过原点最小二乘拟合，报告残差）；物理链（轮周长、挡位比、主减速比，截止到怠速与红线）。可从稳态段自动标定比例 | — | v0.1 |
| `steady_segments` | 按目标车速或转速从加速、变速录音中抽取稳态段：容差内最长连续段取中心，最小时长约束；用于稳态谱与听音片段 | — | v0.1 |
| `rpm_from_ridge` | 无转速计时从信号估计转速：在白化谱图上用阶次梳齿打分（梳齿由缸数、极对数、齿数等先验生成），再用带连续性惩罚的 Viterbi 动态规划选路；可选"移动工况"模式（去除固定频率线）。返回转速、置信度和触界标志 | Peeters et al. 2019（综述）；梳齿与动态规划设计参考基频跟踪文献（Hermes 1988） | v0.2 |
| `virtual_rpm` | 由车速生成虚拟转速曲线 `clip(k·v, idle, redline)`，用于听音或驾驶实验中声音与动力学的解耦 | — | v0.3 |
| `align_signals` | 两套未同步采集系统的时间对齐：拍手等瞬态起始检测与广义互相关（GCC-PHAT）粗对齐，低频波形亚样本精修；起止两次标记估计时钟漂移；报告不确定度 | Knapp & Carter 1976 | v0.4 |

### 2.2 转角域（`rota.angle`）

| 函数 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `angular_resample` | 等时间采样 → 等转角采样（计算阶次跟踪 COT 的核心）。按最小转速自适应的抗混叠低通；高阶插值（样条或多相）；返回角域信号与每转采样数 | Fyfe & de Munck 1997；Brandt 2011 | v0.1 |
| `angular_to_time` | 角域信号映射回等时间采样（角域处理后的逆变换，用于重合成） | 同上 | v0.3 |
| `stationarize` | 已知转速比例（固定传动比、线性升速）时的时间扭曲，使阶次频率恒定，便于长窗谱指标；提供逆变换 | — | v0.3 |

### 2.3 图谱（`rota.maps`）

| 函数 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `rpm_frequency_map` | 频率–转速图（Campbell 图） | Brandt 2011 | v0.1 |
| `rpm_order_map` | 阶次–转速图（基于 `angular_resample`） | Fyfe & de Munck 1997；Brandt 2011 | v0.1 |
| 分箱选项 | 按时间帧，或按 Δrpm 分箱（含迟滞）；单调段选择（升速、降速、自动） | Brandt 2011 | v0.4 |

### 2.4 谱与切片（`rota.slices`）

| 函数 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `order_spectrum` | 阶次图按时间做能量平均，得到平均阶次谱 | Brandt 2011 | v0.1 |
| `order_slice` | 指定阶次的幅值随转速变化（可返回复幅值，供 `OrderTable` 使用） | Brandt 2011 | v0.1 |
| `frequency_slice` | 固定频率切片，用于观察结构共振 | Brandt 2011 | v0.4 |
| `overall_level` | 总声级随转速变化，可选 A 计权 | IEC 61672-1 | v0.4 |
| `metric_vs_rpm` | 任意时变量（心理声学指标、声级、第三方工具输出）沿转速或车速分箱：单调段选择、分箱宽度、每箱最少样本、起始跳过、聚合方式（均值、百分位、功率平均）；返回计数与离散度 | — | v0.4 |
| `frequency_to_order`、`order_share` | 频率轨迹（例如 tonality 工具给出的主导音频率）→ 阶次归属（给定容差）；统计落在某阶及其谐波上的比例 | — | v0.4 |

### 2.5 阶次提取（`rota.extract`，原 `rota.vkf`）

| 函数 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `order_waveform` | Vold-Kalman 二代滤波提取阶次时域波形与复包络。联合多阶（带状对称正定方程，用 SciPy 带状求解器精确求解）；−3 dB 带宽 → 权重映射；多根轴各自的转速参考 | Vold & Leuridan 1993、1995；Vold et al. 1997；Feldbauer & Höldrich 2000；Tůma 2005；Pan & Lin 2006 | v0.2 |
| `tracked_harmonics` | 相位锁定的短时谐波提取：按 `rpm_to_phase` 给出的相位对每阶解调，用满足 COLA 条件的归一化窗做短时平均得到复系数轨迹，再叠加重调制回时域。可选显著性门限（χ² 检验）、转速计延迟。计算量远小于 Vold-Kalman，适合长录音与多通道 | Antoni 2009；角–时循环平稳分析（Antoni 等，待核对） | v0.2 |
| `remove_orders` | 从原信号中减去指定阶次（任一提取器），用于"去掉某阶再听" | 同上 | v0.2 |
| `residual` | 原信号减去所有轴、所有阶次，得到宽带残差 | 同上 | v0.2 |
| `resolution_check` | 可分辨性诊断：相邻阶次或双轴近似同速时的拍频与窗长倒数、包络带宽的关系；等带宽时域窗在低转速下跨越的阶次数；解调带宽是否超过相邻阶次间距。给出警告与建议的窗长（以圈数计）或带宽 | — | v0.2 |
| `kalman_smoother`（可选） | 随机游走包络模型的 Kalman/RTS 平滑器，作为 Vold-Kalman 的贝叶斯参照实现 | — | v0.4 之后 |

### 2.6 重合成（`rota.resynth`，新增）

目标：把分析结果（阶次表、提取的阶次波形、残差）变回可听的声音，并允许修改后重新合成。典型用途：听音实验刺激（"去掉某阶再听"、"只保留宽带"）、虚拟工况（把阶次表映射到新的转速曲线）、驾驶模拟器的声音引擎（块式合成）。

| 函数或类 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `OrderTable` | 数据结构：阶次 × 转速 → 复幅值（Pa），显式标记峰值或 RMS；支持非整数与负阶次、每阶的固定频率偏移（载波边带，如逆变器开关频率的边带）；可由 `order_slice` 或 `tracked_harmonics` 构建；NPZ/JSON 读写 | — | v0.3 |
| `synthesize_orders` | 由 `OrderTable` 和 `rpm(t)` 做加法合成：相位为 2π·阶次·∫rpm/60 dt 的连续积分；在转速网格之间插值的是幅值，不是实部与虚部；超过 Nyquist 的阶次静音而非截断；起始相位可随机（seedable）、共享或取自表中相位 | McAulay & Quatieri 1986 | v0.3 |
| `OrderSynth`（流式） | 块式 API：保持相位累加器与上一块的状态，每次接收一块转速或负载并输出一块音频，供实时程序或模拟器调用（ROTA 本身不做音频 I/O） | — | v0.3 |
| `synthesize_broadband` | 由幅度谱包络（可随转速或车速变化的查表）生成宽带噪声：随机相位 IFFT 颗粒 + 功率互补窗叠加（√Hann，50 % 重叠），无帧率泵动；颗粒内去除超低频；相邻工况之间按 √ 权重交叉淡化 | Serra & Smith 1990；Allen & Rabiner 1977 | v0.3 |
| `remap_rpm` | 把阶次表或提取结果映射到新的转速曲线（虚拟加速、改变传动比、改变录音的表观车速）。这是改变表观车速的正确做法，不是通用变调或变速；每 km/h 的频率变化只有约 0.25 %，转速与频率精度必须远高于此 | — | v0.3 |
| `modify_orders` | 对提取的系数轨迹或阶次表做逐阶增益、静音、频移、替换，再重调制 | — | v0.3 |
| `resynthesize` | 阶次 + 残差重新组合，可指定阶次–残差能量比（dB）；同时返回 orders-only、residual-only、mix 三路 | — | v0.3 |
| `stationary_excerpt` | 从加速录音生成频率恒定、幅值平稳的听音片段：角域平均谱（或正弦 + 残差模型）+ 随机相位 IFFT；音调分量可选真正弦（无拍频）或随机相位簇两种模型；报告包络摆动与峰值因子 | — | v0.3 |
| `provenance` | 为每次重合成写 JSON 侧车：全部参数、种子、ROTA 版本、输入文件摘要 | — | v0.3 |

### 2.7 测试信号（`rota.synth`）

| 函数 | 作用 | 版本 |
|---|---|---|
| `synth_runup` | 合成升速或降速信号。运动学链：车速曲线 → 车轮转速 → 挡位比与主减速比 → 转速；曲线可选线性、幂律、端点零斜率的 S 形或任意数组；怠速、换挡事件与转速回落。阶次以数据给出：阶次（含半阶次）、基准幅值、随转速的幅值律（起始斜坡、最低频率掩模、可选 AM/FM）、初始相位（seedable）；可选共振（传递函数）与固定频率成分；路噪与风噪随车速的包络；多根轴（独立转速、固定转速比、近似同速）；转速计脉冲（每转多脉冲、抖动、丢脉冲）。输出 mix 与各分量 stem（同一增益），以及真值：rpm(t)、转角、每阶幅值与相位、噪声包络，外加 JSON 配置与种子 | v0.1 |
| `rpm_profiles` | 转速曲线生成：线性、指数、分段保持（如 40→100→40 km/h 的映射）、双轴（独立、锁定比例、小差异） | v0.1 |
| `synth_tone_noise` | 听音实验刺激：纯音或谐波复合 + 宽带或驼峰噪声，按总功率分配法设定音噪比（TNR）；锚点信号（如 60 dB SPL 的纯音与白噪） | v0.3 |

### 2.8 验证指标（`rota.metrics`，新增）

| 函数 | 作用 | 版本 |
|---|---|---|
| `track_error` | 转速轨迹误差：中位相对误差、在 3 % 与 5 % 内的帧比例、触界统计 | v0.1 |
| `waveform_similarity` | 相关系数、NRMSE；多通道时按能量加权（避免对低能量通道做均匀平均）；交叉捕获（估计与另一源真值的相关）；零基线（原始混合与真值的相关） | v0.2 |
| `order_residual_energy` | 去阶后残差中目标阶次的剩余能量（dB，相对原始阶次能量） | v0.2 |
| `window_checks` | COLA 与功率互补检验；调制谱中帧率处的峰值（泵动检测） | v0.3 |
| `envelope_stats` | 包络摆动（百分位）、峰值因子、与原始谱的误差 | v0.3 |

### 2.9 心理声学桥接（`rota.tonal`，薄层，新增）

ROTA 不实现响度、尖锐度、粗糙度、ECMA-418-2 tonality 等已有开源实现的指标，只提供桥接：

| 函数 | 作用 | 依据文献 | 版本 |
|---|---|---|---|
| `a_weighting`、`level_db` | IEC 61672-1 A/C 计权（模拟极点双线性变换，1 kHz 归一化），只施加一次（时域单向或频域），不用零相位双向滤波；dB re 20 µPa；RMS 定义显式；双耳能量平均 | IEC 61672-1 | v0.4 |
| `to_calibrated_48k` | 保持校准的多相重采样，输出 48 kHz 的 Pa 信号，供 SQAT、MoSQITo、ECMA-418-2 开源实现使用 | — | v0.4 |
| `order_prominence` | 在阶次跟踪谱（角域）上计算 ECMA-74 Annex D 的音噪比 TNR 与突出比 PR：逐音检测、突出标志、频率相关阈值；窗长足以分辨低频音（如 140 Hz 的音落在约 100 Hz 宽的临界带内）；得到"阶次突出度随转速"曲线 | ECMA-74；ISO 7779；ECMA-418-1 | v0.4 |
| `components` | 阶次分量描述结构（阶次、转速范围、级别、带宽、谐波性指数），便于与 tonality 工具的分量输出对照 | Becker & Sottek 2019（待核对） | v0.4 |

### 2.10 绘图（`rota.plot`）

绘图辅助函数，与计算分离：Campbell 图、阶次图、切片、转速轨迹与置信度、重合成前后对比。v0.4。

---

## 3. 版本计划

**v0.1 核心链路**：完成一次带转速计的加速工况分析。
`tacho_to_rpm`、`rpm_to_phase`、`regularize_control`、`speed_to_rpm`、`steady_segments`、`angular_resample`、`rpm_frequency_map`、`rpm_order_map`、`order_spectrum`、`order_slice`、`synth_runup`、`rpm_profiles`、`track_error`。验证用例 T1、T2、T5、T14。

**v0.2 阶次提取与无转速计**：
`rpm_from_ridge`、`order_waveform`、`tracked_harmonics`、`remove_orders`、`residual`、`resolution_check`、`waveform_similarity`、`order_residual_energy`。验证用例 T3、T4、T6、T8、T10、T12、T13。

**v0.3 重合成**：
`rota.resynth` 全部、`angular_to_time`、`stationarize`、`virtual_rpm`、`synth_tone_noise`、`window_checks`、`envelope_stats`。验证用例 T9、T11。

**v0.4 应用层与桥接**：
`frequency_slice`、`overall_level`、`metric_vs_rpm`、`order_share`、Δrpm 分箱、`align_signals`、`rota.tonal`、`rota.plot`。提供与 SQAT、MoSQITo、ECMA-418-2 开源实现衔接的示例，例如"去掉某阶后响度或 tonality 随转速的变化"。

**v1.0 正式发布**：
完整验证报告、英文文档、PyPI 发布、Zenodo DOI。投稿一篇验证论文（JOSS 或声学会议）。

**暂不做**：

- 实时音频 I/O（只提供块式合成 API）、GUI；
- 转子动力学专用分析（轴心轨迹、波德图）；
- 空间音频（DoA、波束形成、Ambisonics 解码与双耳渲染）：作为独立的配套包，ROTA 只保证多通道线性算子（设计原则 6）；
- 任何专有或商业文件格式的读取（第 4 节）；
- 响度、尖锐度、粗糙度、ECMA-418-2 tonality 等指标本体。

---

## 4. 独立实现规范

ROTA 的 API 对 MATLAB 用户友好，但实现完全独立。所有贡献者遵守以下规则。

### 禁止

1. 阅读、复制或翻译任何商业软件（MATLAB 工具箱、商用 NVH 分析与仿真软件）的源代码或内部数据结构。包括用 `edit`、`type` 打开内置函数，以及在调试器中单步进入内置函数。
2. 复制商业软件文档的文字、示例代码和示例数据。docstring、教程和示例全部自行撰写。示例数据由 `synth_runup` 生成，或来自权利清晰的自有测量。
3. 实现或内置任何专有、未公开文件格式的解析器，也不在文档中描述其内部结构。用户自行把数据转换为 WAV、NPZ、HDF5 或 CSV 后再使用 ROTA。
4. 移植许可不明的第三方代码，包括同事或合作者分享的研究代码。只能依据其发表的论文重新实现，并引用论文。
5. 在项目名、包名或宣传中使用 MATLAB 等商标。
6. 在代码注释中写"复现某软件的某函数"。应写"基于某文献实现"。
7. 把受限数据（合作方车辆录音、被试数据、未公开数据集）作为测试夹具或示例。仓库中不出现车型、合作单位、被试信息和本机绝对路径。

### 允许

1. 依据公开文献实现算法，并在 docstring 的 References 节注明出处。
2. 把 MATLAB 当作黑盒参考：只调用公开函数，记录输入和输出，用于数值对照。
3. 在文档中提供"MATLAB 用户对照表"，事实性地说明功能对应关系。
4. 复用许可兼容的开源代码（如 BSD、MIT），在 NOTICE 文件中致谢；GPL 代码只有在 ROTA 采用 GPL-3.0 时才可复用。

### 流程

- PR 模板包含一项确认："本次提交未参考任何商业软件的源代码，未移植许可不明的代码。"
- 参考或复用第三方开源代码前，先确认许可证与 ROTA 兼容，并在 NOTICE 文件中致谢。
- 每次发布前对整个 git 历史做敏感词检索（商业软件名、合作方、车型、人名、本机路径）。

### 独立性声明（放在 README）

> ROTA 是独立的学术项目，与 MathWorks 及任何商业软件厂商无关，也未获其认可或支持。MATLAB 是 The MathWorks, Inc. 的注册商标。

---

## 5. 验证与对照

### 5.1 原则

- **解析真值是第一标准，MATLAB 是第二参考。** 对照的目的是确认两者对同一物理量的估计一致，而不是逐位复制 MATLAB 的输出。两者不一致时，以真值为准，并在报告中解释差异来源。
- **显式指定参数。** MATLAB 调用中显式给出全部参数（窗函数、分辨率、重叠率、幅值类型、滤波带宽等），不依赖默认值。这样对照可以复现，ROTA 也不需要复制 MATLAB 的默认设置。
- **比较物理量，不比较网格。** 两边的分帧细节可能不同。因此对照沿阶次脊线的幅值、峰值位置、转速曲线等派生量，不逐个比较图谱单元格。
- **重合成用往返测试。** 合成 → 分析 → 与输入表比较；宽带合成用自检（窗功率和、调制谱）。

### 5.2 对照表

| ROTA | MATLAB（Signal Processing Toolbox） | 对照方式 | 对照指标 |
|---|---|---|---|
| `tacho_to_rpm` | `tachorpm` | 直接对照 | 转速曲线误差 |
| `rpm_from_ridge` | `rpmtrack` | 直接对照 | 转速曲线误差；触界标志 |
| `angular_resample` | 无公开对应 | 解析真值；通过 `rpm_order_map` 间接对照 | 转角域中恒定阶次的谱峰宽度和泄漏 |
| `rpm_frequency_map` | `rpmfreqmap` | 直接对照 | 各阶沿脊线的幅值；频率轴 |
| `rpm_order_map` | `rpmordermap` | 直接对照 | 各阶幅值；阶次分辨率 |
| `order_spectrum` | `orderspectrum` | 直接对照 | 峰值阶次和幅值 |
| `order_slice` | `ordertrack`（输入为阶次图） | 直接对照 | 幅值–转速曲线 |
| `order_waveform`（波形） | `orderwaveform` | 直接对照 | 波形相关系数；−3 dB 带宽 |
| `order_waveform`（包络） | `ordertrack`（输入为信号，Vold-Kalman 方式） | 直接对照 | 包络幅值 |
| `tracked_harmonics` | 无公开对应 | 解析真值；与 `order_waveform` 交叉 | 波形相关系数；系数轨迹幅值 |
| `remove_orders`、`residual` | 无公开对应 | 间接：原信号减去 `orderwaveform` 的结果 | 残差中该阶的剩余能量 |
| `resolution_check` | 无 | 按定义（T8） | 警告是否触发 |
| `frequency_slice` | 无公开对应 | 解析真值 | 共振峰位置和幅值 |
| `overall_level`、`rota.tonal` | 不对照 MATLAB | 与 SQAT 的 `sound_level_meter` 交叉验证（开源，已按 IEC 61672-1 验证） | 声级差；TNR/PR 差 |
| `rota.resynth` | 无公开对应 | 往返测试（T9）；自检（T11） | 表往返误差；泵动峰值 |
| `rota.metrics`、`synth_runup` | 无 | 按定义 | — |

### 5.3 测试信号

| 编号 | 内容 | 主要考察 |
|---|---|---|
| T1 | 单阶、恒定幅值、线性加速 | 幅值与转速的绝对精度 |
| T2 | 多阶 + 固定频率共振 + 噪声 | 图谱、切片、共振识别 |
| T3 | 指数减速（非线性转速） | 重采样和脊线跟踪 |
| T4 | 两根轴，阶次交叉 | Vold-Kalman 多阶解耦 |
| T5 | 转速计每转多脉冲、脉冲抖动、丢失脉冲 | `tacho_to_rpm` 的鲁棒性 |
| T6 | 高转速变化率的快速升速 | COT 和 Vold-Kalman 的极限 |
| T7（可选） | 自有实测数据 | 实际工况；仅在数据权利允许时公开 |
| T8 | 双轴转速比接近 1（相差 0.5 %、2 %、4 %），阶次相同 | 可分辨性极限；`resolution_check` 必须在不可分辨时警告 |
| T9 | 阶次表往返：`OrderTable` → `synthesize_orders` → `rpm_order_map` 或 `order_slice` → 表 | 重合成与分析链路的一致性 |
| T10 | 真值转速在 `rpm_from_ridge` 搜索范围之外 | 必须报告触界，不得静默返回边界值 |
| T11 | `synthesize_broadband` 在恒定谱包络下的输出 | 调制谱在帧率处无峰；长时功率恒定 |
| T12 | T2 信号经 `remove_orders` | 残差中目标阶次的剩余能量 |
| T13 | 参考转速含误差（偏置 0.5 %、随机抖动、阶跃）时各提取器的退化曲线 | 提取对转速误差的鲁棒性；文档给出可接受的转速误差 |
| T14 | 含怠速的低转速起步，相邻阶次仅差 0.5 阶 | 阶次分辨率随转速的变化；角域 STFT 与等带宽时域 STFT 的对比 |

### 5.4 初始容差（实测后调整）

| 指标 | 对解析真值 | 对 MATLAB |
|---|---|---|
| 转速中位相对误差 | ≤ 0.1 % | ≤ 0.1 % |
| 无转速计转速：在 3 % 内的帧比例（稳态或缓慢升速） | ≥ 90 % | 与 `rpmtrack` 相差 ≤ 5 个百分点 |
| 平稳段阶次幅值 | ≤ 0.2 dB | ≤ 0.5 dB |
| 共振附近阶次幅值 | ≤ 1 dB | ≤ 1 dB |
| 阶次波形相关系数（去掉首尾） | ≥ 0.99 | ≥ 0.99 |
| Vold-Kalman −3 dB 带宽 | 与设定值相差 ≤ 5 % | 与 MATLAB 相差 ≤ 5 % |
| 阶次表往返幅值（T9） | ≤ 0.2 dB | — |
| 去阶残差中目标阶次剩余能量（T12） | ≤ −20 dB | — |
| 宽带合成帧率泵动（T11） | ≤ 1 dB | — |

### 5.5 目录与流程

```
validation/
├── make_signals.py      # 用 synth_runup 生成 T1–T14 和解析真值，保存为 .mat / .npz
├── signals/             # 测试信号与解析真值
├── matlab/
│   └── run_reference.m  # 只包含：加载数据、调用公开函数、保存输出
├── reference/           # MATLAB 输出，附 MATLAB 与工具箱版本信息
└── compare.py           # 运行 ROTA，与真值和 MATLAB 对比，生成报告
```

1. 运行 `make_signals.py`，生成测试信号和解析真值。
2. 在 MATLAB 中运行 `run_reference.m`。输出存入 `reference/`，同时记录 `ver` 给出的版本信息。
3. 运行 `compare.py`，生成对照报告（表格和图）。
4. 持续集成只运行 pytest（对解析真值）和对已存参考数据的比较，不需要安装 MATLAB。

公开 `reference/` 中的 MATLAB 输出之前，先确认所在机构的 MATLAB 许可条款允许。

---

## 6. 设计约束：来自既往工作的教训

以下经验来自作者既往的阶次分析、听音实验刺激制作和声音合成工作，决定了上文若干函数的设计。ROTA 不复用那些代码，只吸收教训。

**转速与相位**

1. 转速估计器必须报告是否触碰搜索边界。曾有基频跟踪器在整批文件上锁死在搜索下限，后续的角域重采样实际上什么也没做，而流程没有任何报错。→ `rpm_from_ridge` 的触界标志，T10。
2. 单一基频无法对齐多个不相关的谐波族（如电机阶次与逆变器边带）。→ 多轴、多载波的设计：`OrderTable` 的频率偏移，`tracked_harmonics` 与 `order_waveform` 按轴分别处理。
3. 三缸机的整数阶与点火阶（1.5 阶）能量相当，只用点火阶梳齿会锁到 2/3 或 4/3 倍转速；快速升速时窗内阶次频率的变化超过白化带宽，会把阶次本身抹掉。→ 梳齿可配置，白化带宽随工况调整，Viterbi 连续性惩罚。
4. 换挡使转速非单调，沿转速分箱只对固定挡位工况有意义。→ `metric_vs_rpm` 的单调段选择与计数输出。
5. 车速与转速的单位陷阱（m/s 与 km/h、rad/s 与 rpm）反复出错。→ 函数签名中显式单位，文档写明。

**角域与提取**

6. 线性插值的角域重采样没有抗混叠；静态高斯阶次掩模不能跟随幅相调制。→ `angular_resample` 带自适应低通与高阶插值；提取用相位锁定解调或 Vold-Kalman。
7. 双轴转速相差 0.5 % 时，任何单通道跟踪器都会遇到偏差–方差墙（拍频与包络变化率同量级）。→ `resolution_check` 给出可分辨性警告，T8。
8. 多通道指标做均匀平均会严重低估（能量集中在少数通道）。→ `waveform_similarity` 的能量加权。
9. 时间平均谱会抹平漂移的阶次；纯正弦重合成则丢掉噪声底色，听起来不像原录音。→ `stationarize` 或角域平均 + 阶次与残差分开建模。

**重合成**

10. 在转速网格之间插值复幅值的实部与虚部时，相位不同的相邻点会相消，幅值塌陷可达数十 dB，并产生数十 Hz 的斩波。→ 插值幅值，相位单独处理。
11. 随机相位的独立帧叠加必须用功率互补窗（√Hann）；用幅值 COLA 窗会产生帧率的功率泵动（曾达 +14 dB）。颗粒内的超低频会在窗切换处产生咔嗒声。→ `synthesize_broadband` 的窗设计与颗粒高通，T11。
12. 对整谱加增益后再做 RMS 归一化，增益会被抵消。→ 增益只作用于阶次相对残差的超出部分，归一化基于参考合成。
13. 相近的随机相位谱线会产生拍频；而把谱峰合并为高斯簇又会使音调变成窄带噪声、包络大幅摆动。→ `stationary_excerpt` 提供真正弦与随机相位簇两种模型，并报告包络摆动。
14. 幅值约定不明确（峰值或 RMS、每 bin 幅值或功率）会造成 3 dB 乃至 30 dB 的错误。→ `OrderTable` 显式标记，设计原则 8。
15. 参数在笔记本与脚本之间漂移，最终刺激所用的设置无法重建。→ `provenance` 侧车、由文件名派生的确定性种子。
16. 超过 Nyquist 的阶次被钳到 fs/2 会产生一个假音。→ 静音。

**桥接**

17. 第三方心理声学工具需要校准的 Pa、48 kHz、规则时间轴和带缺口的 rpm(t)；它们各自有起始瞬态（响度与 tonality 约 0.6 s，粗糙度约 1.5 s，波动度 4–5 s），聚合时需跳过。→ `to_calibrated_48k`、`regularize_control`、`metric_vs_rpm` 的起始跳过参数。
18. 长窗谱指标（TNR、PR、单帧 Aures）对阶次漂移敏感，听觉模型类指标不敏感。→ `order_prominence` 在角域（阶次恒定）上计算。

**测试信号与验证**

19. 既往的合成测试信号让转速从 0 开始且各阶满幅，幅值与转速无关，并用"合成所用的精确转速 + 已知阶次列表"做验证，属于循环论证。→ `synth_runup` 保存每阶幅值与相位的真值，带怠速与幅值律；验证加入转速误差（T13）和低转速起步（T14）。
20. 等带宽的时域 STFT 重标为阶次轴，在低转速下主瓣跨越多个阶次，半阶次分不开。→ `rpm_order_map` 用角域 STFT，窗长按圈数定义；`resolution_check` 给出提示。
21. 同步解调用固定 20 Hz 低通，相邻阶次（0.5 阶间隔）互相泄漏，转速接近 0 时所有阶次塌缩。→ 解调带宽按阶次间距随转速缩放；`tracked_harmonics` 的窗长以圈数计。
22. 同一算法的 MATLAB 与 Python 实现在采样点数、窗函数、积分规则上悄悄不一致，结果无法互相印证。→ 单一实现、单一真值生成器、配置落盘。

**测量与对齐**

23. 多套采集系统无硬件同步时只能靠拍手，精度为几十 ms；一次录音没有标记就无法对齐。→ `align_signals` 支持起止两次标记与漂移估计，并报告不确定度；文档建议每段录音首尾各一次标记。
24. 慢速通道（车速、加速度）与声学通道的采样率成整数比（如 5:1），且各车的通道布局与标定不同。→ `regularize_control` 接受任意采样率；标定因子显式传入。
25. 球谐基的约定（ACN/SN3D、Condon–Shortley 相位、N3D 缩放）在不同代码间不一致，会让方向图旋转 180° 或按阶加权。这是配套空间包的教训；ROTA 核心只保证按通道线性，不内置任何球谐约定。

**声音操控**

26. 改变录音的表观车速不能靠视频变速或通用的变调、变速；要把阶次与残差分开，按新转速重调制阶次，按车速律缩放残差。每 km/h 的频率变化只有约 0.25 %，转速与频率精度必须远高于此。→ `remap_rpm`、`resynthesize`。
27. 分量分别渲染再混合时的增益必须显式记录（曾有未说明的 ×2 和混音工具的 1/N 缩放）；A 计权用零相位双向滤波会加权两次。→ 设计原则 8；`a_weighting` 只施加一次。

---

## 7. 参考文献

标"待核对"的条目在实现对应函数前核对题录。

- Brandt, A. (2011). *Noise and Vibration Analysis: Signal Analysis and Experimental Procedures*. Wiley.
- Fyfe, K. R., & de Munck, E. D. S. (1997). Analysis of computed order tracking. *Mechanical Systems and Signal Processing*, 11(2), 187–205. doi:10.1006/mssp.1996.0056
- Potter, R. (1990). A new order tracking method for rotating machinery. *Sound and Vibration*, 24(9), 30–34.（待核对）
- Gade, S., Herlufsen, H., Konstantin-Hansen, H., & Wismer, N. J. (1995). Order tracking analysis. *Brüel & Kjær Technical Review*, No. 2.（待核对）
- Vold, H., & Leuridan, J. (1993). High resolution order tracking at extreme slew rates, using Kalman tracking filters. SAE Technical Paper 931288.
- Vold, H., & Leuridan, J. (1995). High resolution order tracking using Kalman tracking filters: theory and applications. SAE Technical Paper 951332.
- Vold, H., Mains, M., & Blough, J. (1997). Theoretical foundations for high performance order tracking with the Vold-Kalman tracking filter. SAE Technical Paper 972007.
- Feldbauer, C., & Höldrich, R. (2000). Realization of Vold-Kalman tracking filter: a least squares problem. *Proceedings of the COST G-6 Conference on Digital Audio Effects (DAFx-00)*, Verona.
- Tůma, J. (2005). The passband width of the Vold-Kalman order tracking filter. *Sborník vědeckých prací VŠB-TU Ostrava, Řada strojní*, 51(2), 149–154.
- Pan, M.-C., & Lin, Y.-F. (2006). Further exploration of Vold–Kalman-filtering order tracking with shaft-speed information—I: Theoretical part, numerical implementation and parameter investigations. *Mechanical Systems and Signal Processing*, 20(5), 1134–1154.
- Pan, M.-C., & Wu, C.-X. (2007). Adaptive Vold–Kalman filtering order tracking. *Mechanical Systems and Signal Processing*, 21(8), 2957–2969.（待核对）
- Peeters, C., Leclère, Q., Antoni, J., Lindahl, P., Donnal, J., Leeb, S., & Helsen, J. (2019). Review and comparison of tacholess instantaneous speed estimation methods on experimental vibration data. *Mechanical Systems and Signal Processing*, 129, 407–436. doi:10.1016/j.ymssp.2019.02.031
- Antoni, J. (2009). Cyclostationarity by examples. *Mechanical Systems and Signal Processing*, 23(4), 987–1036.
- Antoni, J., et al. (2022). 角–时循环平稳框架下的谐波跟踪与提取。*AIAA Journal*, 60.（待核对）
- Rémond, D., Antoni, J., & Randall, R. B. (2014). Editorial for the special issue on instantaneous angular speed (IAS) processing and angular applications. *Mechanical Systems and Signal Processing*, 44.（待核对）
- Hermes, D. J. (1988). Measurement of pitch by subharmonic summation. *Journal of the Acoustical Society of America*, 83(1), 257–264.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 34(4), 744–754.
- Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: A sound analysis/synthesis system based on a deterministic plus stochastic decomposition. *Computer Music Journal*, 14(4), 12–24.
- Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time Fourier analysis and synthesis. *Proceedings of the IEEE*, 65(11), 1558–1564.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
- Knapp, C. H., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 24(4), 320–327.
- Griffin, D. W., & Lim, J. S. (1984). Signal estimation from modified short-time Fourier transform. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 32(2), 236–243.
- Becker, J., & Sottek, R. (2019). Psychoacoustic order tonality calculation. SAE Technical Paper.（待核对）
- Sottek, R., Becker, J., & Lobato, T. (2025). High-resolution option for the Sottek hearing model tonality enables accurate order analysis. SAE Technical Paper.（待核对）
- Dupré, T., et al. (2023). Analysis-by-synthesis of engine sounds. *Acta Acustica*, 7.（待核对）
- IEC 61672-1:2013. Electroacoustics – Sound level meters – Part 1: Specifications.
- ECMA-74 (Annex D: prominent discrete tones); ISO 7779 (Annex D); ECMA-418-1; ECMA-418-2:2025.
- ISO 532-1:2017. Acoustics – Methods for calculating loudness – Part 1: Zwicker method.

---

## 8. 作者与致谢

ROTA 由 Zhenxian Li 设计、实现与维护，是唯一贡献者。开发过程中使用了 Claude（Anthropic）作为辅助工具，用于文献梳理、代码审查和文档整理；所有算法选择、实现与验证由作者负责。
