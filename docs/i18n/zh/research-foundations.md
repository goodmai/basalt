# Basalt 的数学与科学研究基础

> Languages: [English](../../research-foundations.md) · [Русский](../ru/research-foundations.md) · **中文** · [Español](../es/research-foundations.md) · [Français](../fr/research-foundations.md) · [日本語](../ja/research-foundations.md) · [Deutsch](../de/research-foundations.md) · [한국어](../ko/research-foundations.md) · [Bahasa Indonesia](../id/research-foundations.md)

Basalt 站在全球开放科学共同体的肩膀之上。Basalt 中的每一个 Metal 内核、量化编解码器、KV 缓存压缩器、投机采样验证器、认知门控（epistemic gate）以及后训练目标函数，均扎根于国际科研生态中经过同行评审的数学理论与开放获取预印本——横跨**北美、欧洲、中国、俄罗斯、韩国及日本**的学术与工程研究学派。

---

## 致谢与敬意

我们向所有研究人员、数学家、系统工程师、高校实验室以及开源维护者致以最诚挚的感谢与深切的敬意——正是他们的理论突破、严谨证明与参考实现使 Basalt 成为可能：

- **Apple Machine Learning Research 与 MLX 团队**（*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert*，以及 [`ml-explore/mlx`](https://github.com/ml-explore/mlx)、[`mlx-swift`](https://github.com/ml-explore/mlx-swift) 和 [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) 的贡献者）——感谢统一内存数组计算、Metal 自定义内核原语以及 Apple Neural Engine Transformer 研究。
- **康奈尔大学 RelaxML 实验室**（*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*）——感谢 $\mu$-非相干性（$\mu$-incoherence）理论、`QuIP#` $E_8$ Gosset 格编码以及 `QTIP` 移位网格（bitshift trellis）量化。
- **华为计算系统实验室（CSL）、诺亚方舟实验室及 MindIE 团队**（*Lukas Muller, Yuxuan Sun, Ruizhou Ding* 及合作者）——感谢 `MACaQ`、`SINQ`、`KVarN`、`FlatQuant`、`EAQuant` 以及 `MXAttention` 微缩放（microscaling）公式体系。
- **IST Austria、Yandex Research、HSE、MIPT、莫斯科国立大学（MSU）及俄罗斯科学院斯捷克洛夫数学研究所（MIAN）**（*Dan Alistarh, Elias Frantar, Vage Egiazarian, Denis Kuznedelev, Ruslan Svirschevski, Boris Kashin, Ivan Oseledets, D. R. Poimanov, D. N. Davydova* 及合作者）——感谢 `GPTQ`、`AQLM`、`PV-Tuning`、过完备基 LLM 量化、路由器加权 MoE Hessian 矩阵以及双层混合精度搜索。
- **清华大学、DeepSeek-AI、月之暗面（Moonshot AI）、上海交通大学及理想汽车（Li Auto）**（*张锦涛（Jintao Zhang）、陈建飞（Jianfei Chen）、朱军（Jun Zhu）、唐浩天（Haotian Tang）、刘知建（Zhijian Liu）、韩松（Song Han）、冷静文（Jingwen Leng）、过敏意（Minyi Guo）* 及合作者）——感谢 `SageAttention`（`1 / 2++ / 3`）、`Quest`、`DuoAttention`、原生稀疏注意力（`NSA`）、`MoBA`、`LIRQuant` 以及 `ParoQuant`。
- **Google Research、Google DeepMind、耶鲁大学、斯坦福大学、MIT 与加州大学伯克利分校**（*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero* 及合作者）——感谢香农率失真 KV 量化（`QJL`、`PolarQuant`、`TurboQuant`）、基于最优传输的多草稿投机验证（`SpecTr`）、`DiLoCo`、`LeJEPA` 以及 `Harness²`。
- **日本 NLP 与 AI 研究界（`NLP2026`、`JSAI2026`、东京科学大学、电气通信大学、同志社大学、东京大学、AIST、NII LLMC、MBZUAI）**（*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura* 及合作者）——感谢自回归偏好优化（`ADPO` / `cADPO`）、GQA 分组级因果路径修补、反引导分析与残差 KV 量化。
- **韩国 AI 与系统研究界（`KIISE`、`KICS`、中央大学、江原大学、汉阳大学、首尔市立大学、Nota AI）**（*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng* 及合作者）——感谢跨语言 `HeadKV` 不变性、`MAO-SNMF` 多义特征分解、负载感知动态投机树、`FLRot` 以及级联稳定的 MoE 量化。
- **`ggml` / `llama.cpp` 与 Hugging Face 开源社区**（*Georgi Gerganov, Iwan Kawrakow*、`mlx-community` 以及全球开源权重模型作者）——感谢 `K`-quant / `IQ` 重要性网格、`safetensors`、`swift-transformers` 以及开放权重科学研究。

---

## 1. 局部与仿射基变换（`W4A4` / `MXFP4` / `W4A16`）

现代训练后量化通过直接融合进 Metal `simdgroup` 前导代码的 Walsh–Hadamard 正交旋转、Kronecker 分解一般线性共轭以及稀疏 Givens 旋转，抑制重尾权重与激活离群值（$\max_i |x_i| / \|x\|_2$）：

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller 等（华为 CSL） | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | 严格局部的层内分块 Walsh–Hadamard 基 $H_B \in \{H_{32}, H_{64}\}$ + 在 OCP `MXFP4`（`E2M1` + `E8M0`）中进行 Hessian 坐标下降**翻转精修（Flip-Refinement）**，无需跨层旋转约束 | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun 等（华为诺亚方舟 / 清华大学 / 香港中文大学） | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | 将 Stiefel 正交性 $Q \in O(d)$ 松弛为可逆 Kronecker 因子 $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$，将共轭计算复杂度从 $O(d^2)$ 降至 $O(d(\sqrt{d_1}+\sqrt{d_2}))$ | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** 与 **ParoQuant** — X. Xu 等（理想汽车）；Z. Liu 等 | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | 在硬件并行的不相交通道对中施加保持 $L_2$ 范数的局部等距 / 成对 Givens 旋转 $G(i, j, \theta)$ + 逐通道缩放 | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** 与 **FLRot** — A. Makni 等；B. Kang 等（汉阳大学） | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | 面向边缘端与低比特 LLM 的结构化低参数量 / 全可学习离群值抑制旋转 | `#547`, `#554`, `#563` |

---

## 2. 双随机 Sinkhorn 归一化与无校准矩阵条件数优化

根据 Sinkhorn 定理，交替执行行/列缩放可使非负权重幅值同时沿两个坐标轴趋向均匀方差，从而在不依赖校准数据的情况下缩小最坏情况下的量化步长：

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller 等（华为 CSL） | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | 双轴 Sinkhorn–Knopp 归一化 $\widetilde{W} = D_1^{-1} W D_2^{-1}$，在 $O(m n T)$ 复杂度内最小化矩阵不均衡度 $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | 将列缩放因子 $D_2$ 精确代数吸收到前置 `RMSNorm` 参数 $\gamma \leftarrow \gamma \odot d_2$ 中，实现零运行时乘法开销 | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** 与 **EAQuant**（华为诺亚方舟） | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | 激活感知的列目标加权 $c_j \propto \|X_{:,j}\|_2^\alpha$ + MoE 路由器 logit / 专家负载均衡保持 | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. 欧几里得格、网格编码、Babai–LLL 与量化后 `PV-Tuning`

矢量与网格量化器利用高维格（$E_8$、Barnes–Wall、移位寄存器网格）的球形整形增益以及精确最近向量 / 教师 KL 优化：

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng 等（康奈尔 RelaxML） | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | 随机化 Hadamard 非相干变换（`RHT`）+ 寄存器内计算移位网格（`1MAD` / `3INST`），在 `2.0–3.0 bpw` 下逼近高斯率失真极限 | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng 等 | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | `E8P` 2-bit Gosset $E_8$ 格码本（$\mathbb{R}^8$ 中 $256$ 个偶宇称基础壳层向量 $\times 2^8$ 符号 $= 65,536$ 个码字，`2.0 bpw`） | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (`#549`, `#554`) |
| **作为 Babai 最近超平面算法的 GPTQ**、**MR-GPTQ** 与 **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | Cholesky 误差反馈 `GPTQ`（$H = L L^\top$）与 Babai 最近超平面 CVP 算法的等价性；**Lenstra–Lenstra–Lovász (`LLL`)** 格基规约降低正交缺陷；Schur 补 `TurboBoA` 加速精修 | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** 与 **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh 等 | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | 在任意量化表示上交替优化连续缩放/零点参数（`P`）与离散码字分配（`V`）的端到端教师 KL 精修 | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **路由器加权 MoE GPTQ**（`D. R. Poimanov`, MSU）与 **DARTS-QAT**（`D. N. Davydova`, MSU） | *Proc. ISP RAS* 38(6), 2026 · *Intelligent Systems* 30(3), 2026 | MoE 路由器门控 Hessian $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$，在 `Qwen3-30B-A3B` 与 `Mixtral-8x7b` 上达到 `2.8 bpw`（`INT2/INT3`）；双层死区体积约束混合精度 | `GSQRCOConverter`, `#563` |

---

## 4. 信息论 KV 缓存压缩与亚线性上下文检索

为使 32k–131k 思维链（`Long-CoT`）会话常驻内存且不发生内积偏差漂移，Basalt 结合了方差均衡的 2-bit 网格、无偏 Johnson–Lindenstrauss 符号草图、球坐标编码以及保真度校准的注意力头/分页选择：

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — 华为 CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | 在超低比特 KV 量化前进行通道与 token 维度的双轴分块方差均衡 | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**、**PolarQuant** 与 **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni 等 | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | 非对称估计量 $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$ 实现接近香农失真-率界限的**严格零偏差** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$；键向量的极角编码 | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**、**ThinKV** 与 **RQ-KV**（*Watanabe & Taura*, 东京大学） | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | 离群通道混合精度（敏感通道 `4-bit` / 主体通道 `2-bit`）、思考片段自适应压缩（`<think>` 对比最终回答）以及质心残差 KV 量化 | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**、**DuoAttention**、**RazorAttention**、**HeadKV** 与 **跨语言 HeadKV**（*H. Chung 等*, 中央大学） | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | 分页包围盒 top-$k$ 注意力（`Quest`）、基于输出 KL 划分检索头与流式头，以及跨语言检索头不变性（$\text{RBO} \approx 0.8$） | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao 等 | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | 面向非前缀分块组合，在高漂移注意力头上执行选择性 KV 重计算 | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. 硬件对齐的微缩放与可训练块稀疏注意力

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — 华为 MindIE 团队 | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | **统一在线重缩放（`UOS`）**将在线 softmax 的 $m_i$ 更新与 `E8M0` 的 2 的幂次指数移位融合 + **单位分解保持归一化（`PNQ`）**确保量化概率满足 $\sum_j \tilde{P}_{ij} = 1$ | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**、**SageAttention2++** 与 **SageAttention3** — 张锦涛、陈建飞、朱军等（清华大学） | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | 利用 softmax 在键向量按 token 均值中心化 $\tilde{K} = K - \bar{k}$ 下的严格平移不变性：$\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$，在进入 `MXFP4`（`E2M1`）`QK^\top` 与 `PV` 内核前消除通道偏置 | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA**（*原生稀疏注意力*）与 **MoBA**（*块注意力混合*, 月之暗面） | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | 三分支分层注意力（压缩粗粒度摘要 + top-$k$ 细粒度 token 块 + 局部滑动窗口）与无参数质心路由 | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. 输入白化谱分解、Hessian 修复与共享基 MoE

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **SVD-LLM** 与 **SVD-LLM V2** — X. Wang 等 | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | Cholesky 白化 $X X^\top = S S^\top$ 结合 $W S = U \Sigma V^\top$ 的奇异值分解，在激活输出范数下实现精确的 Eckart–Young–Mirsky 最优性：$\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** 与 **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | 跨截断层的顺序累积激活漂移补偿与基于奇异能量曲率的全局秩分配 | `InputWhitenedSVD` (`#552`) |
| **OBR**（*Optimal Brain Restoration*）、**MoE-SVD** 与 **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | 闭式拉格朗日 Hessian 误差补偿 $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$，联合统一剪枝与量化；跨 MoE 专家的共享基 $U_{\text{shared}}$；模态感知的专家保护机制 | `OBRCompensator`, `SharedBasisSwitchLinear`, `MAEPAllocator` (`#552`) |

---

## 7. 无损最优传输投机解码与对比扩散

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **SpecTr**（*最优传输多草稿投机解码*）与 **Gumbel 耦合验证** — Z. Sun, A. T. Suresh 等 | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | 在 $k$ 个候选草稿上构建最优传输耦合，在严格保持精确目标分布 $p_{\text{target}}(x)$（`TV = 0`）的前提下最大化接受率 | `SpecTrAndResidualRaceMetal` (`#160`, `#553`) |
| **Residual Race** 与 **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | 在单次 `simdgroup` 遍历中对残差密度 $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ 执行并行竞速，无需主机往返通信 | `SpecTrAndResidualRaceMetal` (`#553`) |
| **EAGLE-3**（*Y. Li 等*）、**Token Recycling**（*X. Luo 等*）与 **负载感知动态树**（*Y. Chung & T. Shin*） | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | 多层（`low / mid / high`）特征融合草稿头（`fc_fuse`）、基于提示词查表的邻接草稿回收以及热/负载自适应树深 | `EAGLE3FusionHead`, `TokenRecycling`, `SmallMQuantizedLinear` (`#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | 按时间步衰减的 logit 对比 $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$，消除块扩散 / 掩码 LLM 中的重复坍塌 | `DiffuCDProcessor` (`#553`) |

---

## 8. 模型 X 光透视、机制手术、认知逻辑与自监督探针

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **Marchenko–Pastur 谱定律**、**Fisher–Rao 信息几何** 与 **Belnap `SIXTEEN_3` 三格（Trilattice）** — V. A. Marchenko & L. A. Pastur (1967); N. D. Belnap (1977); Y. Shramko & H. Wansing | *Mat. Sb.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | 用于重尾权重诊断的随机矩阵体边缘 $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$；测地线球面距离 $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$；四值次协调真/假/冲突认知门控 | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **拒答特征方向 / 消融手术（Abliteration）**、**CAA 反引导**（*S. Kitada & S. Hara*, UEC）与 **GQA 分组路径修补**（*T. Hara 等*, 同志社大学） | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | 残差流中的线性方向手术、GQA KV 分组因果归因 $I(g)$（$H_g = N_a / N_{kv}$）、认知零空间保护以及 `MAO-SNMF`（*Lee 等*, 2026） | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA**（*Y. LeCun, R. Balestriero 等*）与 **Harness²**（*Google Research*） | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | 无坍塌潜空间预测表征学习（`SIGReg` 各向同性高斯正则化）与递归自验证智能体调度框架 | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. 自回归后训练、分布式优化与 Apple Silicon 协同设计

| 工作 | 发表会议 / 编号 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **ADPO / cADPO**（*S. Oi 等*, Science Tokyo / MBZUAI / AIST / NII LLMC）、**EMA-QLoRA**（*Gaipnazarov & Zhumakulov*）与 **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | 带自适应 $m$ 段对齐的前缀分解自回归偏好损失（AlpacaEval 2 LC 胜率 `40.3% -> 45.8%`）、超复数 Kronecker `PHM` 与正交 `OFT` 适配器 | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**、**零气泡流水线（`ZB-H1`）**、**ZeRO-2** 与 **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | 多台 Mac Thunderbolt 5 / 局域网联邦 Nesterov 外层动量、无气泡反向 `$B/W$` 拆分、分片优化器状态与误差反馈梯度草图压缩 | Distributed Training & Cluster Calibration (`#564`) |
| **Apple Silicon Roofline、ANE Transformers 与核外（Out-of-Core）MoE** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | 统一内存带宽 Roofline 节律控制、GPU + Neural Engine 混合调度与零交换（zero-swap）`POSIX AIO` SSD 专家流式加载 | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. 欧洲（法语区 / 德语区 DACH）认知校准、治理、语义缓存与 UMA 分层（`#621`）

| 工作 | 发表会议 / 生态 | 核心数学原理 | Basalt 子系统 / PR 与 Issue |
|---|---|---|---|
| **Score d'influence 与温度缩放路由** | FR / DACH ML 校准体系 (2025–2026) | 逐注意力头因果影响力评分与经温度校准的路由器置信度门控 | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **领域感知 QAT STE 与量化后审计** | 量化安全性与公平性 (2025–2026) | 直通估计器（STE）领域 QAT 结合质量、公平性与信息泄漏回归审计 | `QATFakeQuantizer`, `QuantizationAuditCore` (`#616`, `#623`) |
| **量化感知 AdaLoRA（`LR/40`）、两阶段 SFT→DPO、自我评估与 RLVR 审计** | 对齐与后训练 (2025–2026) | 量化缩放学习率（`LR/40`）、自适应秩调度、分阶段偏好对齐与捷径利用（shortcut-exploitation）审计 | `QuantizedLRPolicy`, `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (`#617`, `#624`) |
| **LANTERN 语义缓存与 5 组件业务编排器** | 欧洲系统与企业级 AI (2025–2026) | 基于嵌入索引的语义响应复用（带有限过期控制）与多组件企业级路由编排 | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **LAAS-CNRS 神经符号知识核心与 HITL 治理** | LAAS-CNRS / 图卢兹机器人与安全 AI | 符号知识图谱约束结合确定性人机协同（Human-in-the-Loop）审批门控 | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **UMA 感知的 L0/L1/L2 分层 KV 体系与自适应异构规划** | 统一内存系统架构 (2025–2026) | 共享统一内存 L0/L1/L2 KV 预算管理、零拷贝 `L1 → L0` 提升以及异构集群调度 | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
