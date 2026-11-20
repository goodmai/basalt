# Mathematische und wissenschaftliche Forschungsgrundlagen von Basalt

> Languages: [English](../../research-foundations.md) · [Русский](../ru/research-foundations.md) · [中文](../zh/research-foundations.md) · [Español](../es/research-foundations.md) · [Français](../fr/research-foundations.md) · [日本語](../ja/research-foundations.md) · **Deutsch**

Basalt steht auf den Schultern der weltweiten Open-Science-Gemeinschaft. Jeder Metal-Kernel, jeder Quantisierungs-Codec, jeder KV-Cache-Kompressor, jeder spekulative Verifier, jedes epistemische Gate und jede Post-Training-Zielfunktion in Basalt gründet auf begutachteter mathematischer Theorie und Open-Access-Preprints über internationale Forschungsökosysteme hinweg – von **nordamerikanischen, europäischen, chinesischen, russischen, südkoreanischen bis hin zu japanischen** wissenschaftlichen Schulen.

---

## Dank und Anerkennung

Wir drücken allen Forscherinnen und Forschern, Mathematikerinnen und Mathematikern, Systemingenieurinnen und -ingenieuren, Universitätslaboren sowie Open-Source-Maintainerinnen und -Maintainern unseren aufrichtigen Dank und tiefen Respekt aus, deren theoretische Durchbrüche, Beweise und Referenzimplementierungen Basalt möglich gemacht haben:

- **Apple Machine Learning Research & das MLX-Team** (*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert* sowie Mitwirkende an [`ml-explore/mlx`](https://github.com/ml-explore/mlx), [`mlx-swift`](https://github.com/ml-explore/mlx-swift) und [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)) — für Unified-Memory-Array-Computing, Metal-Custom-Kernel-Primitive und Transformer-Forschung auf der Apple Neural Engine.
- **Cornell University RelaxML Lab** (*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*) — für die $\mu$-Inkohärenztheorie, `QuIP#`-$E_8$-Gosset-Gitterkodierung und `QTIP`-Bitshift-Trellis-Quantisierung.
- **Huawei Computing Systems Lab (CSL), Noah's Ark Lab & das MindIE-Team** (*Lukas Muller, Yuxuan Sun, Ruizhou Ding* und Kolleginnen und Kollegen) — für `MACaQ`, `SINQ`, `KVarN`, `FlatQuant`, `EAQuant` und `MXAttention`-Microscaling-Formulierungen.
- **IST Austria, Yandex Research, HSE, MIPT, MSU & Steklov-Institut für Mathematik (MIAN)** (*Dan Alistarh, Elias Frantar, Vage Egiazarian, Denis Kuznedelev, Ruslan Svirschevski, Boris Kashin, Ivan Oseledets, D. R. Poimanov, D. N. Davydova* und Kolleginnen und Kollegen) — für `GPTQ`, `AQLM`, `PV-Tuning`, LLM-Quantisierung mit übervollständigen Basen, router-gewichtete MoE-Hesse-Matrizen und bilevel Mixed-Precision-Suche.
- **Tsinghua-Universität, DeepSeek-AI, Moonshot AI, Shanghai Jiao Tong University & Li Auto** (*Jintao Zhang, Jianfei Chen, Jun Zhu, Haotian Tang, Zhijian Liu, Song Han, Jingwen Leng, Minyi Guo* und Kolleginnen und Kollegen) — für `SageAttention` (`1 / 2++ / 3`), `Quest`, `DuoAttention`, Native Sparse Attention (`NSA`), `MoBA`, `LIRQuant` und `ParoQuant`.
- **Google Research, Google DeepMind, Yale, Stanford, MIT & UC Berkeley** (*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero* und Kolleginnen und Kollegen) — für Shannon-Rate-Distortion-KV-Quantisierung (`QJL`, `PolarQuant`, `TurboQuant`), Optimal-Transport-Multi-Draft-Verifikation (`SpecTr`), `DiLoCo`, `LeJEPA` und `Harness²`.
- **Japanische NLP- & KI-Forschungsgemeinschaft (`NLP2026`, `JSAI2026`, Science Tokyo, UEC, Doshisha Univ., Univ. Tokio, AIST, NII LLMC, MBZUAI)** (*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura* und Kolleginnen und Kollegen) — für Autoregressive Preference Optimization (`ADPO` / `cADPO`), kausales Pfad-Patching auf GQA-Gruppenebene, Anti-Steering-Analyse und residuale KV-Quantisierung.
- **Südkoreanische KI- & System-Community (`KIISE`, `KICS`, Chung-Ang Univ., Kangwon Nat. Univ., Hanyang Univ., Seoul City Univ., Nota AI)** (*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng* und Kolleginnen und Kollegen) — für sprachübergreifende `HeadKV`-Invarianz, polysemantische Feature-Zerlegung `MAO-SNMF`, lastbewusste dynamische Spekulationsbäume, `FLRot` und kaskadenstabile MoE-Quantisierung.
- **Die `ggml`- / `llama.cpp`- & Hugging-Face-Open-Source-Communities** (*Georgi Gerganov, Iwan Kawrakow*, `mlx-community` sowie Autorinnen und Autoren offener Modellgewichte weltweit) — für `K`-Quant- / `IQ`-Wichtigkeitsgitter, `safetensors`, `swift-transformers` und Open-Weight-Wissenschaft.

---

## 1. Lokale und affine Basistransformationen (`W4A4` / `MXFP4` / `W4A16`)

Moderne Post-Training-Quantisierung unterdrückt Ausreißer mit schweren Rändern in Gewichten und Aktivierungen ($\max_i |x_i| / \|x\|_2$) mittels orthogonaler Walsh–Hadamard-Rotationen, Kronecker-faktorisierter allgemeiner linearer Konjugationen und dünnbesetzter Givens-Rotationen, die direkt in Metal-`simdgroup`-Prologe fusioniert werden:

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller et al. (Huawei CSL) | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | Strikt lokale Intra-Layer-Block-Walsh–Hadamard-Basis $H_B \in \{H_{32}, H_{64}\}$ + Hesse-Koordinatenabstieg **Flip-Refinement** in OCP `MXFP4` (`E2M1` + `E8M0`) ohne schichtübergreifende Rotationsnebenbedingungen | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun et al. (Huawei Noah's Ark / Tsinghua / CUHK) | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Lockert die Stiefel-Orthogonalität $Q \in O(d)$ zu invertierbaren Kronecker-Faktoren $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$ und senkt die Konjugationskosten von $O(d^2)$ auf $O(d(\sqrt{d_1}+\sqrt{d_2}))$ | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** & **ParoQuant** — X. Xu et al. (Li Auto); Z. Liu et al. | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Lokale isometrische / paarweise Givens-Rotationen $G(i, j, \theta)$, die die $L_2$-Norm in hardware-parallelen disjunkten Kanalpaaren erhalten, plus kanalweise Skalierung | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** & **FLRot** — A. Makni et al.; B. Kang et al. (Hanyang Univ.) | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | Strukturierte parameterarme bzw. vollständig lernbare ausreißerunterdrückende Rotationen für Edge- und Niedrigbit-LLMs | `#547`, `#554`, `#563` |

---

## 2. Bistochastische Sinkhorn-Normalisierung & kalibrierungsfreie Matrix-Konditionierung

Nach dem Satz von Sinkhorn treibt alternierende Zeilen-/Spaltenskalierung nicht-negative Gewichtsbeträge entlang beider Achsen gleichzeitig auf eine gleichmäßige Varianz zu und verkleinert so die Worst-Case-Quantisierungsschrittweiten ganz ohne Kalibrierungsdaten:

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller et al. (Huawei CSL) | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | Zweiseitige Sinkhorn–Knopp-Normalisierung $\widetilde{W} = D_1^{-1} W D_2^{-1}$, die das Matrix-Ungleichgewicht $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ in $O(m n T)$ minimiert | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | Exakte algebraische Absorption der Spaltenskalierung $D_2$ in das vorangehende `RMSNorm` $\gamma \leftarrow \gamma \odot d_2$ für null Multiplikations-Overhead zur Laufzeit | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** & **EAQuant** (Huawei Noah's Ark) | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | Aktivierungsbewusste Zielgewichtung der Spalten $c_j \propto \|X_{:,j}\|_2^\alpha$ + Erhaltung der MoE-Router-Logits bzw. Expertenbalance | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. Euklidische Gitter, Trellis-Kodierung, Babai–LLL & Post-Quantisierungs-`PV-Tuning`

Vektor- und Trellis-Quantisierer nutzen den sphärischen Formungsgewinn (Shaping Gain) hochdimensionaler Gitter ($E_8$, Barnes–Wall, Schieberegister-Trellis) sowie exakte Nächster-Vektor- / Teacher-KL-Optimierung:

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng et al. (Cornell RelaxML) | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | Randomisierte Hadamard-Inkohärenz (`RHT`) + Compute-in-Register-Bitshift-Trellis (`1MAD` / `3INST`), die bei `2.0–3.0 bpw` nahezu gaußsche Rate-Distortion erreicht | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng et al. | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | `E8P` 2-Bit-Gosset-$E_8$-Gitter-Codebook ($256$ geradparitätische Basisschalen-Vektoren $\times 2^8$ Vorzeichen $= 65,536$ Codewörter in $\mathbb{R}^8$, `2.0 bpw`) | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (`#549`, `#554`) |
| **GPTQ as Babai's Nearest Plane**, **MR-GPTQ** & **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | Äquivalenz von Cholesky-Fehlerfeedback-`GPTQ` ($H = L L^\top$) zu Babais Nächste-Hyperebene-CVP-Algorithmus; **Lenstra–Lenstra–Lovász (`LLL`)**-Gitterbasisreduktion senkt den Orthogonalitätsdefekt; Schur-Komplement-`TurboBoA` beschleunigt die Verfeinerung | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** & **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh et al. | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | Alternierende End-to-End-Teacher-KL-Optimierung über kontinuierliche Skalen/Nullpunkte (`P`) und diskrete Zuweisungen (`V`) auf beliebigen quantisierten Repräsentationen | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **Router-Weighted MoE GPTQ** (`D. R. Poimanov`, MSU) & **DARTS-QAT** (`D. N. Davydova`, MSU) | *Proc. ISP RAS* 38(6), 2026 · *Intelligent Systems* 30(3), 2026 | MoE-routergewichtete Hesse-Matrix $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$, die `2.8 bpw` (`INT2/INT3`) auf `Qwen3-30B-A3B` & `Mixtral-8x7b` erreicht; bilevel größenbeschränkte Mixed-Precision mit Totzone | `GSQRCOConverter`, `#563` |

---

## 4. Informationstheoretische KV-Cache-Kompression & sublinearer Kontextabruf

Um Chain-of-Thought-Sitzungen mit 32k–131k Tokens (`Long-CoT`) ohne Skalarprodukt-Bias-Drift im Speicher zu halten, kombiniert Basalt varianzbalancierte 2-Bit-Gitter, erwartungstreue Johnson–Lindenstrauss-Vorzeichenskizzen, Kugelkoordinaten sowie eine gütekalibrierte Head-/Page-Selektion:

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — Huawei CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | Zweiseitige Varianzbalancierung über Kanal- + Token-Kacheln vor der Ultraniedrigbit-KV-Quantisierung | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**, **PolarQuant** & **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni et al. | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | Asymmetrischer Schätzer $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$ mit **exakt null Bias** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$ nahe Shannons Distortion-Rate-Schranke; Polarwinkelkodierung von Key-Vektoren | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**, **ThinKV** & **RQ-KV** (*Watanabe & Taura*, Univ. Tokio) | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | Ausreißerkanal-Mixed-Precision (`4-bit` empfindliche / `2-bit` Bulk-Kanäle), gedankenspannenadaptive Kompression (`<think>` vs. Antwort) und Zentroid-Residual-KV-Quantisierung | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**, **DuoAttention**, **RazorAttention**, **HeadKV** & **Cross-Lingual HeadKV** (*H. Chung et al.*, Chung-Ang Univ.) | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | Page-Bounding-Box-Top-$k$-Attention (`Quest`), Output-KL-Aufteilung in Retrieval- vs. Streaming-Heads und sprachübergreifende Retrieval-Head-Invarianz ($\text{RBO} \approx 0.8$) | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao et al. | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | Selektive KV-Neuberechnung auf den Attention-Heads mit der größten Drift für nicht-präfixbasierte Chunk-Komposition | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. Hardware-ausgerichtetes Microscaling & trainierbare Block-Sparse-Attention

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — Huawei MindIE Team | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | **Unified Online Rescaling (`UOS`)** zur Fusion von Online-Softmax-$m_i$-Updates mit `E8M0`-Zweierpotenz-Exponentenshifts + **Partition-of-Unity Preserving Normalization (`PNQ`)**, die $\sum_j \tilde{P}_{ij} = 1$ auf quantisierten Wahrscheinlichkeiten garantiert | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**, **SageAttention2++** & **SageAttention3** — J. Zhang, J. Chen, J. Zhu et al. (Tsinghua Univ.) | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | Exakte Translationsinvarianz von Softmax unter Token-Mittelwert-Zentrierung der Keys $\tilde{K} = K - \bar{k}$: $\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$, wodurch Kanal-Bias vor `MXFP4` (`E2M1`) `QK^\top`- & `PV`-Kernels eliminiert wird | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA** (*Native Sparse Attention*) & **MoBA** (*Mixture of Block Attention*, Moonshot AI) | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | Dreizweigige hierarchische Attention (komprimierte grobe Zusammenfassungen + Top-$k$ feine Token-Blöcke + lokales Sliding Window) und parameterfreies Zentroid-Routing | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. Input-Whitened Spektralfaktorisierung, Hesse-Restaurierung & Shared-Basis-MoE

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **SVD-LLM** & **SVD-LLM V2** — X. Wang et al. | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | Cholesky-Whitening $X X^\top = S S^\top$ + SVD von $W S = U \Sigma V^\top$ liefert exakte Eckart–Young–Mirsky-Optimalität in der Aktivierungs-Ausgabenorm $\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** & **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | Sequenzielle Kompensation akkumulierter Aktivierungsdrift über gekürzte Layer hinweg und globale Rangallokation nach Singulärenergie-Krümmung | `InputWhitenedSVD` (`#552`) |
| **OBR** (*Optimal Brain Restoration*), **MoE-SVD** & **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | Geschlossene Lagrange-Hesse-Fehlerkompensation $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$, die Pruning + Quantisierung gemeinsam versöhnt; geteilte Basis $U_{\text{shared}}$ über MoE-Experten; modalitätsbewusster Expertenschutz | `OBRCompensator`, `SharedBasisSwitchLinear`, `MAEPAllocator` (`#552`) |

---

## 7. Verlustfreies Optimal-Transport-Spekulativ-Decoding & kontrastive Diffusion

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **SpecTr** (*Optimal Transport Multi-Draft Speculation*) & **Gumbel-Coupled Verification** — Z. Sun, A. T. Suresh et al. | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | Optimal-Transport-Kopplung über $k$ Kandidaten-Drafts, die die Akzeptanz maximiert und zugleich die exakte Zielverteilung $p_{\text{target}}(x)$ strikt bewahrt (`TV = 0`) | `SpecTrAndResidualRaceMetal` (`#160`, `#553`) |
| **Residual Race** & **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | Paralleles Wettrennen über die Residualdichte $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ in einem einzigen `simdgroup`-Durchlauf ohne Host-Roundtrips | `SpecTrAndResidualRaceMetal` (`#553`) |
| **EAGLE-3** (*Y. Li et al.*), **Token Recycling** (*X. Luo et al.*) & **Load-Aware Dynamic Trees** (*Y. Chung & T. Shin*) | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | Mehrschichtige (`low / mid / high`) Feature-Fusion-Draft-Heads (`fc_fuse`), Prompt-Lookup-Adjazenz-Draft-Recycling sowie thermisch/lastabhängig adaptive Baumtiefe | `EAGLE3FusionHead`, `TokenRecycling`, `SmallMQuantizedLinear` (`#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | Zeitschrittgedämpfter Logit-Kontrast $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$, der Wiederholungskollaps in Block-Diffusion- / maskierten LLMs beseitigt | `DiffuCDProcessor` (`#553`) |

---

## 8. Model-X-Ray, mechanistische Chirurgie, epistemische Logik & selbstüberwachte Sonden

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **Marchenko–Pastur Spectral Law**, **Fisher–Rao Information Geometry** & **Belnap `SIXTEEN_3` Trilattice** — V. A. Marchenko & L. A. Pastur (1967); N. D. Belnap (1977); Y. Shramko & H. Wansing | *Mat. Sb.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | Zufallsmatrix-Bulk-Kante $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$ für Heavy-Tail-Gewichtsdiagnostik; geodätischer Kugeldistanzabstand $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$; vierwertiges parakonsistentes Wahrheits-/Falschheits-/Konflikt-Epistemik-Gating | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **Refusal Feature Direction / Abliteration**, **CAA Anti-Steering** (*S. Kitada & S. Hara*, UEC) & **GQA Group Path Patching** (*T. Hara et al.*, Doshisha Univ.) | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Lineare Richtungschirurgie im Residual-Stream, kausale Attribution auf GQA-KV-Gruppenebene $I(g)$ ($H_g = N_a / N_{kv}$), kognitiver Nullraumschutz und `MAO-SNMF` (*Lee et al.*, 2026) | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA** (*Y. LeCun, R. Balestriero et al.*) & **Harness²** (*Google Research*) | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | Kollapsfreies latentes prädiktives Repräsentationslernen (`SIGReg` isotrope Gauß-Regularisierung) und rekursiv sich selbst verifizierendes Agent-Harnessing | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. Autoregressives Post-Training, verteilte Optimierung & Apple-Silicon-Co-Design

| Arbeit | Publikationsort / ID | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **ADPO / cADPO** (*S. Oi et al.*, Science Tokyo / MBZUAI / AIST / NII LLMC), **EMA-QLoRA** (*Gaipnazarov & Zhumakulov*) & **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | Präfix-faktorisierter autoregressiver Präferenz-Loss mit adaptivem $m$-Segment-Alignment (`40.3% -> 45.8%` AlpacaEval 2 LC Win Rate), hyperkomplexe Kronecker-`PHM` und orthogonale `OFT`-Adapter | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**, **Zero-Bubble Pipeline (`ZB-H1`)**, **ZeRO-2** & **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | Multi-Mac-Thunderbolt-5-/LAN-föderiertes äußeres Nesterov-Momentum, blasenfreie Rückwärts-Aufteilung `$B/W$`, geshardete Optimizer-Zustände und Gradient-Sketching mit Fehlerfeedback | Distributed Training & Cluster Calibration (`#564`) |
| **Apple Silicon Roofline, ANE Transformers & Out-of-Core MoE** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | Unified-Memory-Bandbreiten-Roofline-Taktung, hybrides GPU-+-Neural-Engine-Scheduling und swap-freies `POSIX AIO`-SSD-Experten-Streaming | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. Europäische (FR / DACH) epistemische Kalibrierung, Governance, semantisches Caching & UMA-Tiering (`#621`)

| Arbeit | Publikationsort / Ökosystem | Mathematisches Kernprinzip | Basalt-Subsystem / PRs & Issues |
|---|---|---|---|
| **Score d'influence & Temperature-Scaled Routing** | FR / DACH ML Calibration (2025–2026) | Kausale Einflussbewertung pro Head und temperaturkalibriertes Router-Konfidenz-Gating | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **Domain-Aware QAT STE & Post-Quantization Audit** | Quantization Safety & Fairness (2025–2026) | Domänenspezifisches Straight-Through-Estimator-QAT kombiniert mit Qualitäts-, Fairness- und Leakage-Regressionsaudits | `QATFakeQuantizer`, `QuantizationAuditCore` (`#616`, `#623`) |
| **Quantized AdaLoRA (`LR/40`), Two-Stage SFT→DPO, Self-Appraisal & RLVR Audit** | Alignment & Post-Training (2025–2026) | Quantisierungsskalierte Lernrate (`LR/40`), adaptive Rangplanung, zweistufiges Präferenz-Alignment und Audit auf Shortcut-Ausnutzung | `QuantizedLRPolicy`, `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (`#617`, `#624`) |
| **LANTERN Semantic Caching & 5-Component Business Orchestrator** | European Systems & Enterprise AI (2025–2026) | Embedding-indexierte semantische Antwortwiederverwendung mit begrenzter Veralterung (Bounded Staleness) und mehrkomponentigem Enterprise-Routing | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **LAAS-CNRS Neuro-Symbolic Knowledge Core & HITL Governance** | LAAS-CNRS / Toulouse Robotics & Safety AI | Symbolische Wissensgraph-Constraints gekoppelt mit deterministischen Human-in-the-Loop-Freigabe-Gates | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **UMA-Aware L0/L1/L2 Tiered KV Hierarchy & Adaptive Heterogeneous Planning** | Unified-Memory Systems Architecture (2025–2026) | Gemeinsam genutzte Unified-Memory-L0/L1/L2-KV-Budgetierung mit Zero-Copy-`L1 → L0`-Beförderung und heterogenem Cluster-Scheduling | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
