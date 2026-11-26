# Mathematical & Scientific Research Foundations of Basalt

> Languages: **English** · [Русский](i18n/ru/research-foundations.md) · [中文](i18n/zh/research-foundations.md) · [Español](i18n/es/research-foundations.md) · [Français](i18n/fr/research-foundations.md) · [日本語](i18n/ja/research-foundations.md) · [Deutsch](i18n/de/research-foundations.md) · [한국어](i18n/ko/research-foundations.md) · [Bahasa Indonesia](i18n/id/research-foundations.md)

Basalt stands on the shoulders of the global open-science community. Every Metal kernel, quantization codec, KV-cache compressor, speculative verifier, epistemic gate, and post-training objective in Basalt is grounded in peer-reviewed mathematical theory and open-access preprints across international research ecosystems — spanning **North American, European, Chinese, Russian, South Korean, and Japanese** scientific schools.

---

## Gratitude & Acknowledgements

We express our sincere gratitude and deep respect to all researchers, mathematicians, systems engineers, university laboratories, and open-source maintainers whose theoretical breakthroughs, proofs, and reference implementations made Basalt possible:

- **Apple Machine Learning Research & the MLX Team** (*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert*, and contributors to [`ml-explore/mlx`](https://github.com/ml-explore/mlx), [`mlx-swift`](https://github.com/ml-explore/mlx-swift), and [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)) — for unified-memory array computing, Metal custom-kernel primitives, and Apple Neural Engine transformer research.
- **Cornell University RelaxML Lab** (*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*) — for $\mu$-incoherence theory, `QuIP#` $E_8$ Gosset lattice coding, and `QTIP` bitshift trellis quantization.
- **Huawei Computing Systems Lab (CSL), Noah's Ark Lab & MindIE Team** (*Lukas Muller, Yuxuan Sun, Ruizhou Ding*, and colleagues) — for `MACaQ`, `SINQ`, `KVarN`, `FlatQuant`, `EAQuant`, and `MXAttention` microscaling formulations.
- **IST Austria, Yandex Research, HSE, MIPT, MSU & Steklov Mathematical Institute (MIAN)** (*Dan Alistarh, Elias Frantar, Vage Egiazarian, Denis Kuznedelev, Ruslan Svirschevski, Boris Kashin, Ivan Oseledets, D. R. Poimanov, D. N. Davydova*, and colleagues) — for `GPTQ`, `AQLM`, `PV-Tuning`, overcomplete-basis LLM quantization, router-weighted MoE Hessians, and bilevel mixed-precision search.
- **Tsinghua University, DeepSeek-AI, Moonshot AI, Shanghai Jiao Tong University & Li Auto** (*Jintao Zhang, Jianfei Chen, Jun Zhu, Haotian Tang, Zhijian Liu, Song Han, Jingwen Leng, Minyi Guo*, and colleagues) — for `SageAttention` (`1 / 2++ / 3`), `Quest`, `DuoAttention`, Native Sparse Attention (`NSA`), `MoBA`, `LIRQuant`, and `ParoQuant`.
- **Google Research, Google DeepMind, Yale, Stanford, MIT & UC Berkeley** (*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero*, and colleagues) — for Shannon rate-distortion KV quantization (`QJL`, `PolarQuant`, `TurboQuant`), Optimal-Transport multi-draft speculative verification (`SpecTr`), `DiLoCo`, `LeJEPA`, and `Harness²`.
- **Japanese NLP & AI Research Community (`NLP2026`, `JSAI2026`, Science Tokyo, UEC, Doshisha Univ., Univ. of Tokyo, AIST, NII LLMC, MBZUAI)** (*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura*, and colleagues) — for Autoregressive Preference Optimization (`ADPO` / `cADPO`), GQA group-level causal path patching, anti-steering analysis, and residual KV quantization.
- **South Korean AI & Systems Community (`KIISE`, `KICS`, Chung-Ang Univ., Kangwon Nat. Univ., Hanyang Univ., Seoul City Univ., Nota AI)** (*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng*, and colleagues) — for cross-lingual `HeadKV` invariance, `MAO-SNMF` polysemantic feature decomposition, load-aware dynamic speculative trees, `FLRot`, and cascade-stable MoE quantization.
- **The `ggml` / `llama.cpp` & Hugging Face Open-Source Communities** (*Georgi Gerganov, Iwan Kawrakow*, `mlx-community`, and open-weight model authors worldwide) — for `K`-quant / `IQ` importance grids, `safetensors`, `swift-transformers`, and open-weight science.

---

## 1. Local & Affine Basis Transforms (`W4A4` / `MXFP4` / `W4A16`)

Modern post-training quantization suppresses heavy-tailed weight and activation outliers ($\max_i |x_i| / \|x\|_2$) via orthogonal Walsh–Hadamard rotations, Kronecker-factored general linear conjugations, and sparse Givens rotations fused directly into Metal `simdgroup` prologues:

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller et al. (Huawei CSL) | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | Strictly local intra-layer block Walsh–Hadamard basis $H_B \in \{H_{32}, H_{64}\}$ + Hessian coordinate-descent **Flip-Refinement** in OCP `MXFP4` (`E2M1` + `E8M0`) without cross-layer rotation constraints | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun et al. (Huawei Noah's Ark / Tsinghua / CUHK) | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Relaxes Stiefel orthogonality $Q \in O(d)$ to invertible Kronecker factors $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$, reducing conjugation cost from $O(d^2)$ to $O(d(\sqrt{d_1}+\sqrt{d_2}))$ | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** & **ParoQuant** — X. Xu et al. (Li Auto); Z. Liu et al. | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Local Isometric / Pairwise Givens rotations $G(i, j, \theta)$ preserving $L_2$ norm in hardware-parallel disjoint channel pairs + per-channel scaling | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** & **FLRot** — A. Makni et al.; B. Kang et al. (Hanyang Univ.) | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | Structured low-parameter / fully learnable outlier-suppressing rotations for edge and low-bit LLMs | `#547`, `#554`, `#563` |

---

## 2. Bistochastic Sinkhorn Normalization & Calibration-Free Matrix Conditioning

By Sinkhorn's theorem, alternating row/column scaling drives non-negative weight magnitudes toward uniform variance along both axes simultaneously, shrinking worst-case quantization step sizes without calibration data:

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller et al. (Huawei CSL) | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | Dual-axis Sinkhorn–Knopp normalization $\widetilde{W} = D_1^{-1} W D_2^{-1}$ minimizing matrix imbalance $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ in $O(m n T)$ | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | Exact algebraic absorption of column scale $D_2$ into preceding `RMSNorm` $\gamma \leftarrow \gamma \odot d_2$ for zero runtime multiply overhead | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** & **EAQuant** (Huawei Noah's Ark) | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | Activation-aware column target weighting $c_j \propto \|X_{:,j}\|_2^\alpha$ + MoE router logit / expert-balance preservation | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. Euclidean Lattices, Trellis Coding, Babai–LLL & Post-Quantization `PV-Tuning`

Vector and trellis quantizers exploit the spherical shaping gain of high-dimensional lattices ($E_8$, Barnes–Wall, shift-register trellises) and exact closest-vector / teacher-KL optimization:

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng et al. (Cornell RelaxML) | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | Randomized Hadamard incoherence (`RHT`) + compute-in-register bitshift trellis (`1MAD` / `3INST`) achieving near-Gaussian rate-distortion at `2.0–3.0 bpw` | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng et al. | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | `E8P` 2-bit Gosset $E_8$ lattice codebook ($256$ even-parity base shell vectors $\times 2^8$ signs $= 65,536$ codewords in $\mathbb{R}^8$, `2.0 bpw`) | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (internal kernel primitives, not yet wired to `--to`; `#549`, `#554`) |
| **GPTQ as Babai's Nearest Plane**, **MR-GPTQ** & **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | Equivalence of Cholesky error-feedback `GPTQ` ($H = L L^\top$) to Babai's nearest-hyperplane CVP algorithm; **Lenstra–Lenstra–Lovász (`LLL`)** lattice basis reduction lowers orthogonality defect; Schur-complement `TurboBoA` accelerates refinement | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** & **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh et al. | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | Alternating continuous scale/zero-point (`P`) and discrete assignment (`V`) end-to-end teacher-KL refinement over arbitrary quantized representations | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **Router-Weighted MoE GPTQ** (`D. R. Poimanov`, MSU) & **DARTS-QAT** (`D. N. Davydova`, MSU) | *Proc. ISP RAS* 38(6), 2026 · *Intelligent Systems* 30(3), 2026 | MoE router-gated Hessian $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$ reaching `2.8 bpw` (`INT2/INT3`) on `Qwen3-30B-A3B` & `Mixtral-8x7b`; bilevel dead-zone size-constrained mixed precision | `GSQRCOConverter`, `#563` |

---

## 4. Information-Theoretic KV-Cache Compression & Sub-Linear Context Retrieval

To keep 32k–131k Chain-of-Thought (`Long-CoT`) sessions memory-resident without inner-product bias drift, Basalt combines variance-balanced 2-bit grids, unbiased Johnson–Lindenstrauss sign sketches, spherical coordinates, and fidelity-calibrated head/page selection:

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — Huawei CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | Dual-axis channel + token tile variance balancing prior to ultra-low-bit KV quantization | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**, **PolarQuant** & **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni et al. | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | Asymmetric estimator $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$ achieving **exact zero bias** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$ near Shannon's distortion-rate bound; polar angle coding of key vectors | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**, **ThinKV** & **RQ-KV** (*Watanabe & Taura*, Univ. of Tokyo) | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | Outlier-channel mixed precision (`4-bit` sensitive / `2-bit` bulk), thought-span adaptive compression (`<think>` vs answer), and centroid-residual KV quantization | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**, **DuoAttention**, **RazorAttention**, **HeadKV** & **Cross-Lingual HeadKV** (*H. Chung et al.*, Chung-Ang Univ.) | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | Page-bounding-box top-$k$ attention (`Quest`), output-KL retrieval vs streaming head split, and cross-lingual retrieval-head invariance ($\text{RBO} \approx 0.8$) | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao et al. | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | Selective KV recomputation on top-drift attention heads for non-prefix chunk composition | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. Hardware-Aligned Microscaling & Trainable Block-Sparse Attention

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — Huawei MindIE Team | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | **Unified Online Rescaling (`UOS`)** fusing online softmax $m_i$ updates with `E8M0` power-of-two exponent shifts + **Partition-of-Unity Preserving Normalization (`PNQ`)** guaranteeing $\sum_j \tilde{P}_{ij} = 1$ on quantized probabilities | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**, **SageAttention2++** & **SageAttention3** — J. Zhang, J. Chen, J. Zhu et al. (Tsinghua Univ.) | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | Exact translation invariance of softmax under token-mean key centering $\tilde{K} = K - \bar{k}$: $\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$, eliminating channel bias before `MXFP4` (`E2M1`) `QK^\top` & `PV` kernels | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA** (*Native Sparse Attention*) & **MoBA** (*Mixture of Block Attention*, Moonshot AI) | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | Three-branch hierarchical attention (compressed coarse summaries + top-$k$ fine token blocks + local sliding window) and parameter-free centroid routing | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. Input-Whitened Spectral Factorization, Hessian Restoration & Shared-Basis MoE

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **SVD-LLM** & **SVD-LLM V2** — X. Wang et al. | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | Cholesky whitening $X X^\top = S S^\top$ + SVD of $W S = U \Sigma V^\top$ yields exact Eckart–Young–Mirsky optimality in activation-output norm $\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** & **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | Sequential accumulated activation-drift compensation across truncated layers and global rank allocation by singular-energy curvature | `InputWhitenedSVD` (`#552`) |
| **OBR** (*Optimal Brain Restoration*), **MoE-SVD** & **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | Closed-form Lagrangian Hessian error compensation $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$ jointly reconciling pruning + quantization; shared basis $U_{\text{shared}}$ across MoE experts; modality-aware expert protection | `OptimalBrainRestoration`, `SharedBasisSwitchLinear`, `MAEPExpertScorer` (internal research modules, not yet wired to `--to`; `#552`) |

---

## 7. Lossless Optimal-Transport Speculative Decoding & Contrastive Diffusion

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **SpecTr** (*Optimal Transport Multi-Draft Speculation*) & **Gumbel-Coupled Verification** — Z. Sun, A. T. Suresh et al. | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | Optimal-transport coupling over $k$ candidate drafts maximizing acceptance while strictly preserving exact target distribution $p_{\text{target}}(x)$ (`TV = 0`) | `SpecTrAndResidualRaceMetal` (internal Metal kernel, not yet wired to `serve`; `#160`, `#553`) |
| **Residual Race** & **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | Parallel race over residual density $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ in a single `simdgroup` pass without host round-trips | `SpecTrAndResidualRaceMetal` (internal Metal kernel, not yet wired to `serve`; `#553`) |
| **EAGLE-3** (*Y. Li et al.*), **Token Recycling** (*X. Luo et al.*) & **Load-Aware Dynamic Trees** (*Y. Chung & T. Shin*) | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | Multi-layer (`low / mid / high`) feature-fusion draft head (`fc_fuse`), prompt-lookup adjacency draft recycling, and thermal/load-adaptive tree depth | `TokenRecycling` & `SmallMQuantizedLinear` (wired); `EAGLE3FusionHead` (internal module, not yet wired to `serve`; `#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | Timestep-decayed logit contrast $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$ eliminating repetition collapse in block-diffusion / masked LLMs | `DiffuCDScheduler` (internal research helper, not yet wired to `serve`; `#553`) |

---

## 8. Model X-Ray, Mechanistic Surgery, Epistemic Logic & Self-Supervised Probes

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **Marchenko–Pastur Spectral Law**, **Fisher–Rao Information Geometry** & **Belnap `SIXTEEN_3` Trilattice** — V. A. Marchenko & L. A. Pastur (1967); N. D. Belnap (1977); Y. Shramko & H. Wansing | *Mat. Sb.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | Random-matrix bulk edge $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$ for heavy-tail weight diagnostics; geodesic spherical distance $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$; four-valued paraconsistent truth/falsity/conflict epistemic gating | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **Refusal Feature Direction / Abliteration**, **CAA Anti-Steering** (*S. Kitada & S. Hara*, UEC) & **GQA Group Path Patching** (*T. Hara et al.*, Doshisha Univ.) | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Linear direction surgery in the residual stream, GQA KV-group causal attribution $I(g)$ ($H_g = N_a / N_{kv}$), cognitive-nullspace protection, and `MAO-SNMF` (*Lee et al.*, 2026) | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA** (*Y. LeCun, R. Balestriero et al.*) & **Harness²** (*Google Research*) | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | Collapse-free latent predictive representation learning (`SIGReg` isotropic Gaussian regularization) and recursive self-verifying agent harnessing | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. Autoregressive Post-Training, Distributed Optimization & Apple Silicon Co-Design

| Work | Venue / ID | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **ADPO / cADPO** (*S. Oi et al.*, Science Tokyo / MBZUAI / AIST / NII LLMC), **EMA-QLoRA** (*Gaipnazarov & Zhumakulov*) & **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | Prefix-factored autoregressive preference loss with adaptive $m$-segment alignment (`40.3% -> 45.8%` AlpacaEval 2 LC Win Rate), hypercomplex Kronecker `PHM`, and orthogonal `OFT` adapters | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**, **Zero-Bubble Pipeline (`ZB-H1`)**, **ZeRO-2** & **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | Multi-Mac Thunderbolt 5 / LAN federated Nesterov outer momentum, bubble-free backward `$B/W$` split, sharded optimizer states, and error-feedback gradient sketching | Internal core modules (not yet wired to multi-Mac training; CLI wires `basalt cluster` & `--cluster-peers` calibration, `#564`, `#565`) |
| **Apple Silicon Roofline, ANE Transformers & Out-of-Core MoE** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | Unified-memory bandwidth roofline pacing, GPU + Neural Engine hybrid scheduling, and zero-swap `POSIX AIO` SSD expert streaming | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. European (FR / DACH) Epistemic Calibration, Governance, Semantic Caching & UMA Tiering (`#621`)

| Work | Venue / Ecosystem | Core Mathematical Principle | Basalt Subsystem / PRs & Issues |
|---|---|---|---|
| **Score d'influence & Temperature-Scaled Routing** | FR / DACH ML Calibration (2025–2026) | Per-head causal influence scoring and temperature-calibrated router confidence gating | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **Domain-Aware QAT STE & Post-Quantization Audit** | Quantization Safety & Fairness (2025–2026) | Straight-through-estimator domain QAT paired with quality, fairness and leakage regression audits | `QATFakeQuantizer`, `QuantizationAuditCore` (core/UI primitives, not yet wired inside `UniversalModelConverter`; `#616`, `#623`) |
| **Quantized AdaLoRA (`LR/40`), Two-Stage SFT→DPO, Self-Appraisal & RLVR Audit** | Alignment & Post-Training (2025–2026) | Quantization-scaled learning rate (`LR/40`), adaptive rank scheduling, staged preference alignment and shortcut-exploitation auditing | `QuantizedLRPolicy` (wired on `models train`); `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (internal cores — `--dpo-pairs` and `--adalora-target-rank` refused until wired; offline DPO/GRPO in `#822`) |
| **LANTERN Semantic Caching & 5-Component Business Orchestrator** | European Systems & Enterprise AI (2025–2026) | Embedding-indexed semantic response reuse with bounded staleness and multi-component enterprise routing | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **LAAS-CNRS Neuro-Symbolic Knowledge Core & HITL Governance** | LAAS-CNRS / Toulouse Robotics & Safety AI | Symbolic knowledge graph constraints paired with deterministic Human-in-the-Loop approval gates | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **UMA-Aware L0/L1/L2 Tiered KV Hierarchy & Adaptive Heterogeneous Planning** | Unified-Memory Systems Architecture (2025–2026) | Shared unified-memory L0/L1/L2 KV budgeting with zero-copy `L1 → L0` promotion and heterogeneous cluster scheduling | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
