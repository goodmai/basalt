# Fondements mathématiques et scientifiques de Basalt

> Languages: [English](../../research-foundations.md) · [Русский](../ru/research-foundations.md) · [中文](../zh/research-foundations.md) · [Español](../es/research-foundations.md) · **Français** · [日本語](../ja/research-foundations.md) · [Deutsch](../de/research-foundations.md)

Basalt repose sur les travaux de la communauté internationale de la science ouverte. Chaque noyau Metal, codec de quantification, compresseur de cache KV, vérificateur spéculatif, porte épistémique et objectif de post-entraînement dans Basalt s'appuie sur une théorie mathématique évaluée par les pairs et des prépublications en libre accès issues des écosystèmes de recherche mondiaux — couvrant les écoles scientifiques **nord-américaine, européenne, chinoise, russe, sud-coréenne et japonaise**.

---

## Gratitude et remerciements

Nous exprimons notre sincère gratitude et notre profond respect à tous les chercheurs, mathématiciens, ingénieurs systèmes, laboratoires universitaires et mainteneurs open-source dont les avancées théoriques, les démonstrations et les implémentations de référence ont rendu Basalt possible :

- **Apple Machine Learning Research & l'équipe MLX** (*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert*, et les contributeurs de [`ml-explore/mlx`](https://github.com/ml-explore/mlx), [`mlx-swift`](https://github.com/ml-explore/mlx-swift) et [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)) — pour le calcul matriciel en mémoire unifiée, les primitives de noyaux Metal personnalisés et la recherche sur les transformers pour l'Apple Neural Engine.
- **Cornell University RelaxML Lab** (*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*) — pour la théorie de la $\mu$-incohérence, le codage sur réseau de Gosset $E_8$ `QuIP#` et la quantification par treillis à décalage de bits `QTIP`.
- **Huawei Computing Systems Lab (CSL), Noah's Ark Lab & l'équipe MindIE** (*Lukas Muller, Yuxuan Sun, Ruizhou Ding* et leurs collègues) — pour les formulations de micro-mise à l'échelle `MACaQ`, `SINQ`, `KVarN`, `FlatQuant`, `EAQuant` et `MXAttention`.
- **IST Austria, Yandex Research, HSE, MIPT, MSU & Institut de mathématiques Steklov (MIAN)** (*Dan Alistarh, Elias Frantar, Vage Egiazarian, Denis Kuznedelev, Ruslan Svirschevski, Boris Kashin, Ivan Oseledets, D. R. Poimanov, D. N. Davydova* et leurs collègues) — pour `GPTQ`, `AQLM`, `PV-Tuning`, la quantification de LLM sur bases surcomplètes, les hessiennes MoE pondérées par le routeur et la recherche biniveau en précision mixte.
- **Université Tsinghua, DeepSeek-AI, Moonshot AI, Université Jiao-tong de Shanghai & Li Auto** (*Jintao Zhang, Jianfei Chen, Jun Zhu, Haotian Tang, Zhijian Liu, Song Han, Jingwen Leng, Minyi Guo* et leurs collègues) — pour `SageAttention` (`1 / 2++ / 3`), `Quest`, `DuoAttention`, Native Sparse Attention (`NSA`), `MoBA`, `LIRQuant` et `ParoQuant`.
- **Google Research, Google DeepMind, Yale, Stanford, MIT & UC Berkeley** (*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero* et leurs collègues) — pour la quantification KV débit-distorsion de Shannon (`QJL`, `PolarQuant`, `TurboQuant`), la vérification spéculative multi-brouillons par transport optimal (`SpecTr`), `DiLoCo`, `LeJEPA` et `Harness²`.
- **Communauté japonaise de TAL et d'IA (`NLP2026`, `JSAI2026`, Science Tokyo, UEC, Univ. Doshisha, Univ. de Tokyo, AIST, NII LLMC, MBZUAI)** (*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura* et leurs collègues) — pour l'optimisation de préférences autorégressive (`ADPO` / `cADPO`), le patch de chemins causaux au niveau des groupes GQA, l'analyse anti-pilotage et la quantification KV résiduelle.
- **Communauté sud-coréenne d'IA et de systèmes (`KIISE`, `KICS`, Univ. Chung-Ang, Univ. nat. de Kangwon, Univ. Hanyang, Univ. de Séoul, Nota AI)** (*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng* et leurs collègues) — pour l'invariance interlinguale de `HeadKV`, la décomposition de caractéristiques polysémantiques `MAO-SNMF`, les arbres spéculatifs dynamiques sensibles à la charge, `FLRot` et la quantification MoE stable en cascade.
- **Les communautés open-source `ggml` / `llama.cpp` et Hugging Face** (*Georgi Gerganov, Iwan Kawrakow*, `mlx-community`, et les auteurs de modèles à poids ouverts du monde entier) — pour les grilles d'importance `K`-quant / `IQ`, `safetensors`, `swift-transformers` et la science des modèles ouverts.

---

## 1. Transformations de bases locales et affines (`W4A4` / `MXFP4` / `W4A16`)

La quantification post-entraînement moderne supprime les valeurs aberrantes à queue lourde dans les poids et les activations ($\max_i |x_i| / \|x\|_2$) grâce à des rotations orthogonales de Walsh–Hadamard, des conjugaisons linéaires générales factorisées par Kronecker et des rotations de Givens parcimonieuses fusionnées directement dans les prologues `simdgroup` Metal :

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller et al. (Huawei CSL) | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | Base de Walsh–Hadamard par blocs strictement locale intra-couche $H_B \in \{H_{32}, H_{64}\}$ + **Flip-Refinement** par descente de coordonnées hessienne en OCP `MXFP4` (`E2M1` + `E8M0`) sans contraintes de rotation inter-couches | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun et al. (Huawei Noah's Ark / Tsinghua / CUHK) | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Relâche l'orthogonalité de Stiefel $Q \in O(d)$ en facteurs de Kronecker inversibles $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$, réduisant le coût de conjugaison de $O(d^2)$ à $O(d(\sqrt{d_1}+\sqrt{d_2}))$ | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** & **ParoQuant** — X. Xu et al. (Li Auto); Z. Liu et al. | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Rotations de Givens isométriques locales / par paires $G(i, j, \theta)$ préservant la norme $L_2$ dans des paires de canaux disjointes parallélisables matériellement + mise à l'échelle par canal | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** & **FLRot** — A. Makni et al.; B. Kang et al. (Hanyang Univ.) | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | Rotations structurées à faible nombre de paramètres / entièrement apprenables supprimant les valeurs aberrantes pour les LLM embarqués et bas-bit | `#547`, `#554`, `#563` |

---

## 2. Normalisation bistochastique de Sinkhorn et conditionnement matriciel sans calibration

D'après le théorème de Sinkhorn, l'alternance de mises à l'échelle par ligne et par colonne amène les magnitudes de poids non négatives vers une variance uniforme le long des deux axes simultanément, réduisant les pas de quantification dans le pire cas sans données de calibration :

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller et al. (Huawei CSL) | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | Normalisation bi-axiale de Sinkhorn–Knopp $\widetilde{W} = D_1^{-1} W D_2^{-1}$ minimisant le déséquilibre matriciel $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ en $O(m n T)$ | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | Absorption algébrique exacte de l'échelle colonne $D_2$ dans la `RMSNorm` précédente $\gamma \leftarrow \gamma \odot d_2$ pour un surcoût multiplicatif nul à l'exécution | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** & **EAQuant** (Huawei Noah's Ark) | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | Pondération cible de colonne sensible aux activations $c_j \propto \|X_{:,j}\|_2^\alpha$ + préservation des logits du routeur MoE et de l'équilibre des experts | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. Réseaux euclidiens, codage par treillis, Babai–LLL et post-quantification `PV-Tuning`

Les quantificateurs vectoriels et par treillis exploitent le gain de mise en forme sphérique des réseaux de haute dimension ($E_8$, Barnes–Wall, treillis à registres à décalage) ainsi que l'optimisation exacte du vecteur le plus proche et de la divergence KL par rapport à l'enseignant :

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng et al. (Cornell RelaxML) | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | Incohérence de Hadamard randomisée (`RHT`) + treillis à décalage de bits calculé en registre (`1MAD` / `3INST`) atteignant un rapport débit-distorsion quasi-gaussien à `2.0–3.0 bpw` | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng et al. | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | Dictionnaire sur réseau de Gosset $E_8$ 2 bits `E8P` ($256$ vecteurs de couche de base à parité paire $\times 2^8$ signes $= 65,536$ mots de code dans $\mathbb{R}^8$, `2.0 bpw`) | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (`#549`, `#554`) |
| **GPTQ as Babai's Nearest Plane**, **MR-GPTQ** & **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | Équivalence entre `GPTQ` à rétroaction d'erreur de Cholesky ($H = L L^\top$) et l'algorithme CVP de l'hyperplan le plus proche de Babai ; la réduction de base de réseau **Lenstra–Lenstra–Lovász (`LLL`)** abaisse le défaut d'orthogonalité ; `TurboBoA` par complément de Schur accélère le raffinement | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** & **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh et al. | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | Raffinement KL enseignant de bout en bout alternant échelles/points zéro continus (`P`) et affectation discrète (`V`) sur des représentations quantifiées arbitraires | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **Router-Weighted MoE GPTQ** (`D. R. Poimanov`, MSU) & **DARTS-QAT** (`D. N. Davydova`, MSU) | *Proc. ISP RAS* 38(6), 2026 · *Intelligent Systems* 30(3), 2026 | Hessienne pondérée par le routeur MoE $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$ atteignant `2.8 bpw` (`INT2/INT3`) sur `Qwen3-30B-A3B` & `Mixtral-8x7b` ; précision mixte biniveau avec zone morte sous contrainte de taille | `GSQRCOConverter`, `#563` |

---

## 4. Compression du cache KV issue de la théorie de l'information et récupération sous-linéaire du contexte

Pour conserver en mémoire des sessions Chain-of-Thought de 32k–131k (`Long-CoT`) sans dérive de biais du produit scalaire, Basalt combine des grilles 2 bits équilibrées en variance, des esquisses de signes de Johnson–Lindenstrauss sans biais, des coordonnées sphériques et une sélection de têtes/pages calibrée sur la fidélité :

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — Huawei CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | Équilibrage bi-axial de la variance par tuile (canal + jeton) avant la quantification KV à très bas nombre de bits | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**, **PolarQuant** & **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni et al. | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | Estimateur asymétrique $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$ atteignant un **biais exactement nul** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$ proche de la borne débit-distorsion de Shannon ; codage par angles polaires des vecteurs clés | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**, **ThinKV** & **RQ-KV** (*Watanabe & Taura*, Univ. of Tokyo) | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | Précision mixte selon les canaux aberrants (`4-bit` sensible / `2-bit` masse), compression adaptative selon la portée de réflexion (`<think>` vs réponse), et quantification KV résiduelle par centroïdes | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**, **DuoAttention**, **RazorAttention**, **HeadKV** & **Cross-Lingual HeadKV** (*H. Chung et al.*, Chung-Ang Univ.) | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | Attention top-$k$ par boîte englobante de page (`Quest`), séparation entre têtes de récupération et têtes de streaming par KL de sortie, et invariance interlinguale des têtes de récupération ($\text{RBO} \approx 0.8$) | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao et al. | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | Recalcul sélectif du KV sur les têtes d'attention à plus forte dérive pour la composition de segments hors préfixe | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. Micro-mise à l'échelle alignée sur le matériel et attention parcimonieuse par blocs entraînable

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — Huawei MindIE Team | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | **Ré-échelonnage unifié en ligne (`UOS`)** fusionnant les mises à jour de softmax en ligne $m_i$ avec des décalages d'exposants en puissance de deux `E8M0` + **Normalisation préservant la partition de l'unité (`PNQ`)** garantissant $\sum_j \tilde{P}_{ij} = 1$ sur les probabilités quantifiées | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**, **SageAttention2++** & **SageAttention3** — J. Zhang, J. Chen, J. Zhu et al. (Tsinghua Univ.) | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | Invariance exacte par translation du softmax sous centrage des clés par moyenne de jetons $\tilde{K} = K - \bar{k}$ : $\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$, éliminant le biais de canal avant les noyaux `MXFP4` (`E2M1`) `QK^\top` & `PV` | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA** (*Native Sparse Attention*) & **MoBA** (*Mixture of Block Attention*, Moonshot AI) | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | Attention hiérarchique à trois branches (résumés grossiers compressés + top-$k$ blocs fins de jetons + fenêtre glissante locale) et routage par centroïdes sans paramètres | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. Factorisation spectrale blanchie en entrée, restauration hessienne et MoE à base partagée

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **SVD-LLM** & **SVD-LLM V2** — X. Wang et al. | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | Le blanchiment de Cholesky $X X^\top = S S^\top$ + SVD de $W S = U \Sigma V^\top$ donne l'optimalité exacte d'Eckart–Young–Mirsky en norme de sortie d'activation $\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** & **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | Compensation séquentielle de la dérive d'activation accumulée à travers les couches tronquées et allocation globale du rang selon la courbure de l'énergie singulière | `InputWhitenedSVD` (`#552`) |
| **OBR** (*Optimal Brain Restoration*), **MoE-SVD** & **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | Compensation d'erreur hessienne de Lagrange sous forme fermée $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$ réconciliant conjointement élagage et quantification ; base partagée $U_{\text{shared}}$ entre experts MoE ; protection des experts sensible à la modalité | `OBRCompensator`, `SharedBasisSwitchLinear`, `MAEPAllocator` (`#552`) |

---

## 7. Décodage spéculatif sans perte par transport optimal et diffusion contrastive

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **SpecTr** (*Optimal Transport Multi-Draft Speculation*) & **Gumbel-Coupled Verification** — Z. Sun, A. T. Suresh et al. | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | Couplage par transport optimal sur $k$ brouillons candidats maximisant l'acceptation tout en préservant strictement la distribution cible exacte $p_{\text{target}}(x)$ (`TV = 0`) | `SpecTrAndResidualRaceMetal` (`#160`, `#553`) |
| **Residual Race** & **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | Course parallèle sur la densité résiduelle $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ en une seule passe `simdgroup` sans aller-retour hôte | `SpecTrAndResidualRaceMetal` (`#553`) |
| **EAGLE-3** (*Y. Li et al.*), **Token Recycling** (*X. Luo et al.*) & **Load-Aware Dynamic Trees** (*Y. Chung & T. Shin*) | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | Tête de brouillon à fusion de caractéristiques multicouches (`low / mid / high`, `fc_fuse`), recyclage de brouillons par adjacence dans l'invite, et profondeur d'arbre adaptative à la charge et à la thermique | `EAGLE3FusionHead`, `TokenRecycling`, `SmallMQuantizedLinear` (`#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | Contraste de logits à décroissance temporelle $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$ éliminant l'effondrement répétitif dans les LLM masqués / à diffusion par blocs | `DiffuCDProcessor` (`#553`) |

---

## 8. Radiographie du modèle, chirurgie mécaniste, logique épistémique et sondes auto-supervisées

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **Loi spectrale de Marchenko–Pastur**, **Géométrie de l'information de Fisher–Rao** & **Trilattice `SIXTEEN_3` de Belnap** — V. A. Marchenko & L. A. Pastur (1967); N. D. Belnap (1977); Y. Shramko & H. Wansing | *Mat. Sb.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | Bord du bulk de matrices aléatoires $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$ pour le diagnostic de poids à queue lourde ; distance géodésique sphérique $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$ ; porte épistémique paraconsistante à quatre valeurs vérité/fausseté/conflit | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **Direction de caractéristique de refus / Ablitération**, **Anti-pilotage CAA** (*S. Kitada & S. Hara*, UEC) & **Patch de chemins par groupe GQA** (*T. Hara et al.*, Doshisha Univ.) | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Chirurgie de direction linéaire dans le flux résiduel, attribution causale par groupe KV GQA $I(g)$ ($H_g = N_a / N_{kv}$), protection de l'espace nul cognitif et `MAO-SNMF` (*Lee et al.*, 2026) | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA** (*Y. LeCun, R. Balestriero et al.*) & **Harness²** (*Google Research*) | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | Apprentissage de représentations prédictives latentes sans effondrement (régularisation gaussienne isotrope `SIGReg`) et orchestration d'agents auto-vérifiante récursive | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. Post-entraînement autorégressif, optimisation distribuée et co-conception Apple Silicon

| Travaux | Conférence / ID | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **ADPO / cADPO** (*S. Oi et al.*, Science Tokyo / MBZUAI / AIST / NII LLMC), **EMA-QLoRA** (*Gaipnazarov & Zhumakulov*) & **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | Perte de préférence autorégressive factorisée par préfixe avec alignement adaptatif à $m$ segments (`40.3% -> 45.8%` AlpacaEval 2 LC Win Rate), Kronecker hypercomplexe `PHM` et adaptateurs orthogonaux `OFT` | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**, **Pipeline sans bulles (`ZB-H1`)**, **ZeRO-2** & **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | Momentum externe fédéré de Nesterov sur Thunderbolt 5 / LAN multi-Mac, découpage rétrograde sans bulle `$B/W$`, états d'optimiseur fragmentés et esquisse de gradient à rétroaction d'erreur | Entraînement distribué et calibration de grappe (`#564`) |
| **Modèle Roofline Apple Silicon, Transformers ANE & MoE hors cœur** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | Régulation selon la bande passante de la mémoire unifiée, ordonnancement hybride GPU + Neural Engine et streaming d'experts SSD sans swap via `POSIX AIO` | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. Calibration épistémique européenne (FR / DACH), gouvernance, mise en cache sémantique et hiérarchie UMA (`#621`)

| Travaux | Conférence / Écosystème | Principe mathématique fondamental | Sous-système Basalt / PR & Issues |
|---|---|---|---|
| **Score d'influence et routage calibré en température** | Calibration ML FR / DACH (2025–2026) | Score d'influence causale par tête et filtrage de confiance du routeur calibré en température | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **QAT STE sensible au domaine et audit post-quantification** | Sûreté et équité de la quantification (2025–2026) | QAT par domaine avec estimateur direct (straight-through-estimator) couplé à des audits de régression sur la qualité, l'équité et les fuites | `QATFakeQuantizer`, `QuantizationAuditCore` (`#616`, `#623`) |
| **AdaLoRA quantifié (`LR/40`), SFT→DPO en deux étapes, auto-évaluation et audit RLVR** | Alignement et post-entraînement (2025–2026) | Taux d'apprentissage ajusté à la quantification (`LR/40`), ordonnancement adaptatif du rang, alignement de préférences par étapes et audit d'exploitation de raccourcis | `QuantizedLRPolicy`, `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (`#617`, `#624`) |
| **Cache sémantique LANTERN et orchestrateur métier à 5 composants** | Systèmes européens et IA d'entreprise (2025–2026) | Réutilisation sémantique de réponses indexée par plongements avec obsolescence bornée et routage d'entreprise multi-composants | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **Noyau de connaissances neuro-symbolique LAAS-CNRS et gouvernance HITL** | LAAS-CNRS / Robotique toulousaine et IA de sûreté | Contraintes de graphe de connaissances symbolique associées à des portes déterministes d'approbation humaine dans la boucle (Human-in-the-Loop) | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **Hiérarchie KV à niveaux L0/L1/L2 sensible à l'UMA et planification hétérogène adaptative** | Architecture de systèmes à mémoire unifiée (2025–2026) | Budgétisation KV L0/L1/L2 en mémoire unifiée partagée avec promotion `L1 → L0` sans copie et ordonnancement hétérogène de grappe | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
