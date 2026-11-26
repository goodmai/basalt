# Basalt の数学的・科学的研究基盤

> Languages: [English](../../research-foundations.md) · [Русский](../ru/research-foundations.md) · [中文](../zh/research-foundations.md) · [Español](../es/research-foundations.md) · [Français](../fr/research-foundations.md) · **日本語** · [Deutsch](../de/research-foundations.md) · [한국어](../ko/research-foundations.md) · [Bahasa Indonesia](../id/research-foundations.md)

Basalt は世界のオープンサイエンス・コミュニティの肩の上に立っています。Basalt におけるすべての Metal カーネル、量子化コーデック、KV キャッシュ圧縮器、投機的検証器、認識論的ゲート（epistemic gate）、および事後学習目的関数は、**北米・欧州・中国・ロシア・韓国・日本**の科学研究コミュニティにわたる査読付き数理理論とオープンアクセスプレプリントに裏打ちされています。

---

## 謝辞

理論的なブレークスルー、証明、そして参照実装によって Basalt の実現を可能にしてくださったすべての研究者、数学者、システムエンジニア、大学研究室、およびオープンソースメンテナの皆様に、心よりの感謝と深い敬意を表します:

- **Apple Machine Learning Research & MLX チーム**（*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert*、および [`ml-explore/mlx`](https://github.com/ml-explore/mlx)、[`mlx-swift`](https://github.com/ml-explore/mlx-swift)、[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) の貢献者の皆様）— ユニファイドメモリ配列計算、Metal カスタムカーネルプリミティブ、Apple Neural Engine 向け Transformer 研究。
- **コーネル大学 RelaxML Lab**（*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*）— $\mu$-非干渉性（$\mu$-incoherence）理論、`QuIP#` $E_8$ Gosset 格子符号化、`QTIP` ビットシフトトレリス量子化。
- **Huawei Computing Systems Lab (CSL), Noah's Ark Lab & MindIE チーム**（*Lukas Muller, Yuxuan Sun, Ruizhou Ding* ほか共同研究者の皆様）— `MACaQ`、`SINQ`、`KVarN`、`FlatQuant`、`EAQuant`、および `MXAttention` マイクロスケーリング定式化。
- **IST Austria, Yandex Research, HSE, MIPT, MSU & ステクロフ数学研究所 (MIAN)**（*Dan Alistarh, Elias Frantar, Vage Egiazarian, Denis Kuznedelev, Ruslan Svirschevski, Boris Kashin, Ivan Oseledets, D. R. Poimanov, D. N. Davydova* ほか共同研究者の皆様）— `GPTQ`、`AQLM`、`PV-Tuning`、過剰完備基底 LLM 量子化、ルーター重み付き MoE ヘッセ行列、および二段階混合精度探索。
- **清華大学, DeepSeek-AI, Moonshot AI, 上海交通大学 & Li Auto**（*Jintao Zhang, Jianfei Chen, Jun Zhu, Haotian Tang, Zhijian Liu, Song Han, Jingwen Leng, Minyi Guo* ほか共同研究者の皆様）— `SageAttention` (`1 / 2++ / 3`)、`Quest`、`DuoAttention`、Native Sparse Attention (`NSA`)、`MoBA`、`LIRQuant`、および `ParoQuant`。
- **Google Research, Google DeepMind, イェール大学, スタンフォード大学, MIT & UC バークレー**（*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero* ほか共同研究者の皆様）— Shannon レート歪み KV 量子化（`QJL`、`PolarQuant`、`TurboQuant`）、最適輸送マルチドラフト投機的検証（`SpecTr`）、`DiLoCo`、`LeJEPA`、および `Harness²`。
- **日本の NLP・AI 研究コミュニティ (`NLP2026`, `JSAI2026`, 東京科学大学, 電気通信大学, 同志社大学, 東京大学, 産総研 (AIST), NII LLMC, MBZUAI)**（*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura* ほか共同研究者の皆様）— 自己回帰選好最適化（`ADPO` / `cADPO`）、GQA グループ単位因果パスパッチング、アンチステアリング解析、および残差 KV 量子化。
- **韓国の AI・システム研究コミュニティ (`KIISE`, `KICS`, 中央大学校, 江原大学校, 漢陽大学校, ソウル市立大学校, Nota AI)**（*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng* ほか共同研究者の皆様）— 言語横断 `HeadKV` 不変性、`MAO-SNMF` 多義的特徴量分解、負荷認識型動的投機木、`FLRot`、およびカスケード安定 MoE 量子化。
- **`ggml` / `llama.cpp` & Hugging Face オープンソースコミュニティ**（*Georgi Gerganov, Iwan Kawrakow*、`mlx-community`、および世界中のオープンウェイトモデル著者の皆様）— `K`-quant / `IQ` 重要度グリッド、`safetensors`、`swift-transformers`、およびオープンウェイト科学。

---

## 1. 局所基底変換およびアフィン基底変換 (`W4A4` / `MXFP4` / `W4A16`)

現代の学習後量子化は、直交 Walsh–Hadamard 回転、Kronecker 分解された一般線形共役変換、および Metal の `simdgroup` プロローグへ直接融合される疎な Givens 回転によって、裾の重い重み・活性化外れ値（$\max_i |x_i| / \|x\|_2$）を抑制します:

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller et al. (Huawei CSL) | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | 層間回転制約を必要としない、厳密に局所的な層内ブロック Walsh–Hadamard 基底 $H_B \in \{H_{32}, H_{64}\}$ と OCP `MXFP4` (`E2M1` + `E8M0`) におけるヘッセ行列座標降下 **Flip-Refinement** | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun et al. (Huawei Noah's Ark / 清華大 / CUHK) | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Stiefel 直交性 $Q \in O(d)$ を可逆な Kronecker 因子 $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$ へ緩和し、共役変換の計算量を $O(d^2)$ から $O(d(\sqrt{d_1}+\sqrt{d_2}))$ へ削減 | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** & **ParoQuant** — X. Xu et al. (Li Auto); Z. Liu et al. | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | ハードウェア並列な互いに素なチャネル対で $L_2$ ノルムを保存する局所等長／ペアワイズ Givens 回転 $G(i, j, \theta)$ ＋ チャネルごとのスケーリング | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** & **FLRot** — A. Makni et al.; B. Kang et al. (漢陽大) | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | エッジおよび低ビット LLM 向けの、構造化低パラメータ／完全学習可能な外れ値抑制回転 | `#547`, `#554`, `#563` |

---

## 2. 二重確率 Sinkhorn 正規化とキャリブレーション不要の行列条件付け

Sinkhorn の定理により、行方向と列方向の交互スケーリングを行うことで非負の重み絶対値が両軸に沿って同時に均一な分散へと収束し、キャリブレーションデータなしで最悪ケースの量子化ステップ幅を縮小できます:

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller et al. (Huawei CSL) | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | 計算量 $O(m n T)$ で行列不均衡度 $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ を最小化する二軸 Sinkhorn–Knopp 正規化 $\widetilde{W} = D_1^{-1} W D_2^{-1}$ | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | 列スケール $D_2$ を先行する `RMSNorm` へ $\gamma \leftarrow \gamma \odot d_2$ として代数的に厳密吸収し、実行時の乗算オーバーヘッドをゼロ化 | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** & **EAQuant** (Huawei Noah's Ark) | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | 活性化認識型列目標重み付け $c_j \propto \|X_{:,j}\|_2^\alpha$ ＋ MoE ルーターロジット／エキスパート均衡の保持 | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. ユークリッド格子、トレリス符号化、Babai–LLL、および量子化後 `PV-Tuning`

ベクトル量子化器およびトレリス量子化器は、高次元格子（$E_8$、Barnes–Wall、シフトレジスタトレリス）の球面整形利得と、厳密な最近接ベクトル探索／教師 KL 最適化を活用します:

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng et al. (Cornell RelaxML) | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | ランダム化 Hadamard 非干渉化（`RHT`）＋ レジスタ内計算ビットシフトトレリス（`1MAD` / `3INST`）により、`2.0–3.0 bpw` でガウス分布に近いレート歪みを達成 | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng et al. | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | `E8P` 2 ビット Gosset $E_8$ 格子コードブック（$\mathbb{R}^8$ における $256$ 個の偶パリティ基底シェルベクトル $\times 2^8$ 符号 $= 65,536$ コードワード、`2.0 bpw`） | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (`#549`, `#554`) |
| **GPTQ as Babai's Nearest Plane**, **MR-GPTQ** & **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | コレスキー誤差フィードバック `GPTQ` ($H = L L^\top$) と Babai の最近接超平面 CVP アルゴリズムとの等価性。**Lenstra–Lenstra–Lovász (`LLL`)** 格子基底簡約により直交性欠陥を低減し、Schur 補元による `TurboBoA` でリファインメントを高速化 | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** & **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh et al. | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | 任意の量子化表現に対する連続スケール／ゼロ点（`P`）と離散割り当て（`V`）の交互エンドツーエンド教師 KL リファインメント | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **Router-Weighted MoE GPTQ** (`D. R. Poimanov`, MSU) & **DARTS-QAT** (`D. N. Davydova`, MSU) | *Proc. ISP RAS* 38(6), 2026 · *Intelligent Systems* 30(3), 2026 | `Qwen3-30B-A3B` および `Mixtral-8x7b` で `2.8 bpw` (`INT2/INT3`) に到達する MoE ルーターゲート付きヘッセ行列 $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$、および二段階デッドゾーンサイズ制約付き混合精度 | `GSQRCOConverter`, `#563` |

---

## 4. 情報理論的 KV キャッシュ圧縮と劣線形コンテキスト検索

32k–131k に及ぶ思考の連鎖（`Long-CoT`）セッションを内積バイアスドリフトなしにメモリ常駐させるため、Basalt は分散バランス型 2 ビットグリッド、不偏 Johnson–Lindenstrauss 符号スケッチ、球面座標、および忠実度較正済みヘッド／ページ選択を組み合わせています:

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — Huawei CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | 超低ビット KV 量子化に先立つ、チャネル＋トークン二軸タイルの分散バランシング | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**, **PolarQuant** & **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni et al. | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | Shannon の歪みレート限界近傍で**厳密なゼロバイアス** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$ を達成する非対称推定量 $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$、およびキーベクトルの極角符号化 | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**, **ThinKV** & **RQ-KV** (*Watanabe & Taura*, 東京大) | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | 外れ値チャネル混合精度（高感度 `4-bit` / バルク `2-bit`）、思考スパン適応圧縮（`<think>` 対回答）、およびセントロイド残差 KV 量子化 | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**, **DuoAttention**, **RazorAttention**, **HeadKV** & **Cross-Lingual HeadKV** (*H. Chung et al.*, 中央大) | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | ページバウンディングボックス top-$k$ アテンション（`Quest`）、出力 KL に基づく検索ヘッド対ストリーミングヘッドの分割、および言語横断的な検索ヘッド不変性（$\text{RBO} \approx 0.8$） | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao et al. | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | 非プレフィックスチャンク合成における、ドリフト上位アテンションヘッドでの選択的 KV 再計算 | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. ハードウェア整合マイクロスケーリングと学習可能ブロック疎アテンション

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — Huawei MindIE チーム | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | オンラインソフトマックスの $m_i$ 更新と `E8M0` 2 のべき乗指数シフトを融合する **Unified Online Rescaling (`UOS`)** ＋ 量子化確率上で $\sum_j \tilde{P}_{ij} = 1$ を保証する **Partition-of-Unity Preserving Normalization (`PNQ`)** | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**, **SageAttention2++** & **SageAttention3** — J. Zhang, J. Chen, J. Zhu et al. (清華大) | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | トークン平均キー中心化 $\tilde{K} = K - \bar{k}$ に対するソフトマックスの厳密な並進不変性 $\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$ により、`MXFP4` (`E2M1`) `QK^\top` および `PV` カーネル前にチャネルバイアスを除去 | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA** (*Native Sparse Attention*) & **MoBA** (*Mixture of Block Attention*, Moonshot AI) | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | 3 ブランチ階層アテンション（圧縮された粗い要約＋ top-$k$ の精細トークンブロック＋局所スライディングウィンドウ）とパラメータ不要のセントロイドルーティング | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. 入力白色化スペクトル分解、ヘッセ行列復元、および共有基底 MoE

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **SVD-LLM** & **SVD-LLM V2** — X. Wang et al. | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | コレスキー白色化 $X X^\top = S S^\top$ と $W S = U \Sigma V^\top$ の SVD により、活性化出力ノルム $\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ における厳密な Eckart–Young–Mirsky 最適性を獲得 | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** & **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | 切り詰められたレイヤーを跨ぐ逐次累積活性化ドリフト補償、および特異エネルギー曲率に基づくグローバルランク割り当て | `InputWhitenedSVD` (`#552`) |
| **OBR** (*Optimal Brain Restoration*), **MoE-SVD** & **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | 枝刈りと量子化を統合的に調停する閉形式ラグランジュ・ヘッセ誤差補償 $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$、MoE エキスパート間の共有基底 $U_{\text{shared}}$、およびモダリティ認識型エキスパート保護 | `OptimalBrainRestoration`, `SharedBasisSwitchLinear`, `MAEPExpertScorer` (`#552`) |

---

## 7. ロスレス最適輸送投機的デコーディングと対比拡散

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **SpecTr** (*Optimal Transport Multi-Draft Speculation*) & **Gumbel-Coupled Verification** — Z. Sun, A. T. Suresh et al. | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | $k$ 個の候補ドラフトに対する最適輸送カップリングにより、厳密な目標分布 $p_{\text{target}}(x)$ (`TV = 0`) を完全に保存したまま採択率を最大化 | `SpecTrAndResidualRaceMetal` (`#160`, `#553`) |
| **Residual Race** & **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | ホストラウンドトリップなしに単一の `simdgroup` パス内で残差密度 $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ 上の並列レースを実行 | `SpecTrAndResidualRaceMetal` (`#553`) |
| **EAGLE-3** (*Y. Li et al.*), **Token Recycling** (*X. Luo et al.*) & **Load-Aware Dynamic Trees** (*Y. Chung & T. Shin*) | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | 多層（`low / mid / high`）特徴量融合ドラフトヘッド（`fc_fuse`）、プロンプトルックアップ隣接ドラフト再利用、およびサーマル／負荷適応型の木深さ制御 | `EAGLE3FusionHead`, `TokenRecycling`, `SmallMQuantizedLinear` (`#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | ブロック拡散／マスクド LLM における反復崩壊を排除する時間ステップ減衰ロジット対比 $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$ | `DiffuCDScheduler` (`#553`) |

---

## 8. モデル X-ray、メカニズム手術、認識論理、および自己教師ありプローブ

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **Marchenko–Pastur スペクトル則**, **Fisher–Rao 情報幾何** & **Belnap `SIXTEEN_3` 三格子** — V. A. Marchenko & L. A. Pastur (1967); N. D. Belnap (1977); Y. Shramko & H. Wansing | *Mat. Sb.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | 裾の重い重み診断のためのランダム行列バルク境界 $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$、測地線球面距離 $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$、および 4 値矛盾許容型（真／偽／矛盾）認識論的ゲーティング | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **拒否特徴方向 / Abliteration**, **CAA アンチステアリング** (*S. Kitada & S. Hara*, 電通大) & **GQA グループパスパッチング** (*T. Hara et al.*, 同志社大) | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | 残差ストリームにおける線形方向手術、GQA KV グループ因果帰属 $I(g)$ ($H_g = N_a / N_{kv}$)、認知ヌルスペース保護、および `MAO-SNMF` (*Lee et al.*, 2026) | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA** (*Y. LeCun, R. Balestriero et al.*) & **Harness²** (*Google Research*) | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | 表現崩壊のない潜在予測表現学習（`SIGReg` 等方性ガウス正則化）と再帰的自己検証エージェントハーネス | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. 自己回帰事後学習、分散最適化、および Apple Silicon 協調設計

| 研究 | 発表場所 / ID | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **ADPO / cADPO** (*S. Oi et al.*, 東京科学大 / MBZUAI / AIST / NII LLMC), **EMA-QLoRA** (*Gaipnazarov & Zhumakulov*) & **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | 適応的 $m$ セグメントアラインメントを備えたプレフィックス分解自己回帰選好損失（AlpacaEval 2 LC 勝率 `40.3% -> 45.8%`）、超複素数 Kronecker `PHM`、および直交 `OFT` アダプター | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**, **Zero-Bubble パイプライン (`ZB-H1`)**, **ZeRO-2** & **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | マルチ Mac Thunderbolt 5 / LAN 連合 Nesterov 外側モメンタム、バブルのない逆伝播 `$B/W$` 分割、シャーディングされたオプティマイザ状態、および誤差フィードバック勾配スケッチ | Distributed Training & Cluster Calibration (`#564`) |
| **Apple Silicon ルーフライン, ANE Transformer & アウトオブコア MoE** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | ユニファイドメモリ帯域幅ルーフライン・ペーシング、GPU + Neural Engine ハイブリッドスケジューリング、およびゼロスワップ `POSIX AIO` SSD エキスパートストリーミング | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. 欧州 (FR / DACH) における認識的キャリブレーション、ガバナンス、セマンティックキャッシング、および UMA 階層化 (`#621`)

| 研究 | 発表場所 / エコシステム | 中核となる数学的原理 | Basalt サブシステム / PR & Issue |
|---|---|---|---|
| **Score d'influence & 温度スケーリングルーティング** | FR / DACH ML キャリブレーション (2025–2026) | ヘッドごとの因果影響度スコアリングと温度較正されたルーター信頼度ゲーティング | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **ドメイン認識型 QAT STE & 量子化後監査** | 量子化の安全性と公平性 (2025–2026) | 品質・公平性・リーク回帰監査と組み合わせたストレートスルー推定量（STE）ドメイン QAT | `QATFakeQuantizer`, `QuantizationAuditCore` (`#616`, `#623`) |
| **量子化 AdaLoRA (`LR/40`), 二段階 SFT→DPO, 自己評価 & RLVR 監査** | アラインメントと事後学習 (2025–2026) | 量子化スケールされた学習率（`LR/40`）、適応的ランクスケジューリング、段階的選好アラインメント、およびショートカット悪用（shortcut-exploitation）監査 | `QuantizedLRPolicy`, `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (`#617`, `#624`) |
| **LANTERN セマンティックキャッシング & 5 コンポーネント・ビジネスオーケストレータ** | 欧州システム & エンタープライズ AI (2025–2026) | 有界な鮮度保証（bounded staleness）を備えた埋め込みインデックス付きセマンティック応答再利用、およびマルチコンポーネント・エンタープライズルーティング | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **LAAS-CNRS ニューロシンボリック知識コア & HITL ガバナンス** | LAAS-CNRS / トゥールーズ ロボティクス & 安全 AI | シンボリック知識グラフ制約と決定論的 Human-in-the-Loop 承認ゲートの統合 | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **UMA 認識型 L0/L1/L2 階層化 KV ハイアラーキ & 適応型ヘテロジニアス計画** | ユニファイドメモリシステムアーキテクチャ (2025–2026) | ゼロコピー `L1 → L0` 昇格を伴う共有ユニファイドメモリ L0/L1/L2 KV バジェッティングとヘテロジニアスクラスタースケジューリング | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
