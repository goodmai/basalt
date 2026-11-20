# Fundamentos matemáticos y científicos de la investigación de Basalt

> Languages: [English](../../research-foundations.md) · [Русский](../ru/research-foundations.md) · [中文](../zh/research-foundations.md) · **Español** · [Français](../fr/research-foundations.md) · [日本語](../ja/research-foundations.md) · [Deutsch](../de/research-foundations.md)

Basalt se apoya sobre los hombros de la comunidad global de ciencia abierta. Cada kernel Metal, códec de cuantización, compresor de caché KV, verificador especulativo, puerta epistémica y objetivo de postentrenamiento en Basalt se fundamenta en teoría matemática revisada por pares y preprints de acceso abierto procedentes de ecosistemas científicos internacionales, abarcando las escuelas científicas **norteamericana, europea, china, rusa, surcoreana y japonesa**.

---

## Gratitud y agradecimientos

Expresamos nuestra sincera gratitud y profundo respeto a todos los investigadores, matemáticos, ingenieros de sistemas, laboratorios universitarios y mantenedores de código abierto cuyos avances teóricos, demostraciones e implementaciones de referencia hicieron posible Basalt:

- **Apple Machine Learning Research y el equipo de MLX** (*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert* y colaboradores de [`ml-explore/mlx`](https://github.com/ml-explore/mlx), [`mlx-swift`](https://github.com/ml-explore/mlx-swift) y [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)) — por la computación de arreglos en memoria unificada, primitivas de kernels personalizados en Metal y la investigación de transformers en el Apple Neural Engine.
- **Cornell University RelaxML Lab** (*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*) — por la teoría de $\mu$-incoherencia, la codificación en retículo de Gosset $E_8$ de `QuIP#` y la cuantización trellis por desplazamiento de bits `QTIP`.
- **Huawei Computing Systems Lab (CSL), Noah's Ark Lab y el equipo MindIE** (*Lukas Muller, Yuxuan Sun, Ruizhou Ding* y colegas) — por `MACaQ`, `SINQ`, `KVarN`, `FlatQuant`, `EAQuant` y las formulaciones de microescalado `MXAttention`.
- **IST Austria, Yandex Research, HSE, MIPT, MSU e Instituto Matemático Steklov (MIAN)** (*Dan Alistarh, Elias Frantar, Vage Egiazarian, Denis Kuznedelev, Ruslan Svirschevski, Boris Kashin, Ivan Oseledets, D. R. Poimanov, D. N. Davydova* y colegas) — por `GPTQ`, `AQLM`, `PV-Tuning`, cuantización de LLM con bases sobrecompletas, hessianos MoE ponderados por router y búsqueda binivel de precisión mixta.
- **Universidad de Tsinghua, DeepSeek-AI, Moonshot AI, Universidad Jiao Tong de Shanghái y Li Auto** (*Jintao Zhang, Jianfei Chen, Jun Zhu, Haotian Tang, Zhijian Liu, Song Han, Jingwen Leng, Minyi Guo* y colegas) — por `SageAttention` (`1 / 2++ / 3`), `Quest`, `DuoAttention`, Native Sparse Attention (`NSA`), `MoBA`, `LIRQuant` y `ParoQuant`.
- **Google Research, Google DeepMind, Yale, Stanford, MIT y UC Berkeley** (*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero* y colegas) — por la cuantización KV basada en la función tasa-distorsión de Shannon (`QJL`, `PolarQuant`, `TurboQuant`), verificación especulativa multiborrador mediante transporte óptimo (`SpecTr`), `DiLoCo`, `LeJEPA` y `Harness²`.
- **Comunidad japonesa de PLN e IA (`NLP2026`, `JSAI2026`, Science Tokyo, UEC, Univ. Doshisha, Univ. de Tokio, AIST, NII LLMC, MBZUAI)** (*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura* y colegas) — por la optimización autorregresiva de preferencias (`ADPO` / `cADPO`), parcheo de rutas causales a nivel de grupo GQA, análisis anti-steering y cuantización KV residual.
- **Comunidad surcoreana de IA y sistemas (`KIISE`, `KICS`, Univ. Chung-Ang, Univ. Nac. de Kangwon, Univ. Hanyang, Univ. de la Ciudad de Seúl, Nota AI)** (*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng* y colegas) — por la invariancia translingüe de `HeadKV`, descomposición de características polisémicas `MAO-SNMF`, árboles especulativos dinámicos conscientes de la carga, `FLRot` y cuantización MoE estable en cascada.
- **Las comunidades de código abierto `ggml` / `llama.cpp` y Hugging Face** (*Georgi Gerganov, Iwan Kawrakow*, `mlx-community` y autores de modelos de pesos abiertos en todo el mundo) — por las mallas de importancia `K`-quant / `IQ`, `safetensors`, `swift-transformers` y la ciencia abierta de modelos.

---

## 1. Transformaciones de base locales y afines (`W4A4` / `MXFP4` / `W4A16`)

La cuantización moderna posterior al entrenamiento suprime los valores atípicos de cola pesada en pesos y activaciones ($\max_i |x_i| / \|x\|_2$) mediante rotaciones ortogonales de Walsh–Hadamard, conjugaciones lineales generales factorizadas por Kronecker y rotaciones dispersas de Givens fusionadas directamente en los prólogos `simdgroup` de Metal:

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller et al. (Huawei CSL) | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | Base de bloques Walsh–Hadamard estrictamente local intracapa $H_B \in \{H_{32}, H_{64}\}$ + descenso por coordenadas hessiano **Flip-Refinement** en OCP `MXFP4` (`E2M1` + `E8M0`) sin restricciones de rotación entre capas | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun et al. (Huawei Noah's Ark / Tsinghua / CUHK) | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Relaja la ortogonalidad de Stiefel $Q \in O(d)$ a factores de Kronecker invertibles $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$, reduciendo el coste de conjugación de $O(d^2)$ a $O(d(\sqrt{d_1}+\sqrt{d_2}))$ | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** y **ParoQuant** — X. Xu et al. (Li Auto); Z. Liu et al. | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Rotaciones isométricas locales / por pares de Givens $G(i, j, \theta)$ que preservan la norma $L_2$ en pares de canales disjuntos paralelos en hardware + escalado por canal | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** y **FLRot** — A. Makni et al.; B. Kang et al. (Hanyang Univ.) | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | Rotaciones estructuradas de pocos parámetros / totalmente aprendibles para suprimir valores atípicos en LLM de borde y baja precisión | `#547`, `#554`, `#563` |

---

## 2. Normalización biestocástica de Sinkhorn y acondicionamiento matricial sin calibración

Por el teorema de Sinkhorn, el escalado alternado de filas y columnas lleva las magnitudes no negativas de los pesos hacia una varianza uniforme en ambos ejes simultáneamente, reduciendo los pasos de cuantización en el peor caso sin necesidad de datos de calibración:

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller et al. (Huawei CSL) | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | Normalización de Sinkhorn–Knopp de doble eje $\widetilde{W} = D_1^{-1} W D_2^{-1}$ que minimiza el desequilibrio matricial $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ en $O(m n T)$ | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | Absorción algebraica exacta de la escala de columna $D_2$ en la `RMSNorm` precedente $\gamma \leftarrow \gamma \odot d_2$ para lograr sobrecarga cero de multiplicaciones en tiempo de ejecución | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** y **EAQuant** (Huawei Noah's Ark) | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | Ponderación objetivo de columnas consciente de activaciones $c_j \propto \|X_{:,j}\|_2^\alpha$ + preservación de logits del router MoE y equilibrio de expertos | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. Retículos euclídeos, codificación trellis, Babai–LLL y `PV-Tuning` posterior a la cuantización

Los cuantizadores vectoriales y trellis aprovechan la ganancia de conformación esférica de retículos de alta dimensión ($E_8$, Barnes–Wall, trellises de registro de desplazamiento) y la optimización exacta de vector más cercano y KL respecto al profesor:

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng et al. (Cornell RelaxML) | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | Incoherencia de Hadamard aleatorizada (`RHT`) + trellis por desplazamiento de bits en registros (`1MAD` / `3INST`) que alcanza una tasa-distorsión cercana a la gaussiana en `2.0–3.0 bpw` | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng et al. | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | Libro de códigos de retículo de Gosset $E_8$ de 2 bits `E8P` ($256$ vectores de capa base de paridad par $\times 2^8$ signos $= 65,536$ palabras código en $\mathbb{R}^8$, `2.0 bpw`) | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (`#549`, `#554`) |
| **GPTQ as Babai's Nearest Plane**, **MR-GPTQ** y **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | Equivalencia de `GPTQ` con realimentación de error de Cholesky ($H = L L^\top$) respecto al algoritmo CVP de hiperplano más cercano de Babai; la reducción de base reticular **Lenstra–Lenstra–Lovász (`LLL`)** reduce el defecto de ortogonalidad; el complemento de Schur `TurboBoA` acelera el refinamiento | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** y **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh et al. | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | Optimización alternada de escala/punto cero continuos (`P`) y asignación discreta (`V`) de extremo a extremo mediante KL del profesor sobre representaciones cuantizadas arbitrarias | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **Router-Weighted MoE GPTQ** (`D. R. Poimanov`, MSU) y **DARTS-QAT** (`D. N. Davydova`, MSU) | *Proc. ISP RAS* 38(6), 2026 · *Intelligent Systems* 30(3), 2026 | Hessiano ponderado por router MoE $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$ que alcanza `2.8 bpw` (`INT2/INT3`) en `Qwen3-30B-A3B` y `Mixtral-8x7b`; precisión mixta binivel de zona muerta con restricción de tamaño | `GSQRCOConverter`, `#563` |

---

## 4. Compresión de caché KV basada en teoría de la información y recuperación de contexto sublineal

Para mantener residentes en memoria sesiones Chain-of-Thought (`Long-CoT`) de 32k–131k sin deriva de sesgo en el producto interno, Basalt combina mallas de 2 bits equilibradas en varianza, bosquejos de signo no sesgados de Johnson–Lindenstrauss, coordenadas esféricas y selección de cabezas/páginas calibrada por fidelidad:

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — Huawei CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | Equilibrado de varianza en bloques bidimensionales de canal + token antes de la cuantización KV de bits ultrabajos | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**, **PolarQuant** y **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni et al. | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | Estimador asimétrico $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$ que logra **sesgo exactamente cero** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$ cerca de la cota tasa-distorsión de Shannon; codificación de ángulos polares para vectores clave | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**, **ThinKV** y **RQ-KV** (*Watanabe & Taura*, Univ. of Tokyo) | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | Precisión mixta para canales atípicos (`4-bit` sensible / `2-bit` grueso), compresión adaptativa por tramos de pensamiento (`<think>` frente a respuesta) y cuantización KV por residuo de centroides | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**, **DuoAttention**, **RazorAttention**, **HeadKV** y **Cross-Lingual HeadKV** (*H. Chung et al.*, Chung-Ang Univ.) | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | Atención top-$k$ con caja delimitadora de página (`Quest`), división de cabezas de recuperación frente a streaming mediante KL de salida e invariancia translingüe de cabezas de recuperación ($\text{RBO} \approx 0.8$) | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao et al. | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | Recálculo selectivo de KV en cabezas de atención de mayor deriva para composición de fragmentos sin prefijo común | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. Microescalado alineado al hardware y atención dispersa por bloques entrenable

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — Huawei MindIE Team | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | **Reescalado en línea unificado (`UOS`)** que fusiona actualizaciones de softmax en línea $m_i$ con desplazamientos de exponente potencia de dos `E8M0` + **Normalización preservadora de partición de la unidad (`PNQ`)** que garantiza $\sum_j \tilde{P}_{ij} = 1$ sobre probabilidades cuantizadas | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**, **SageAttention2++** y **SageAttention3** — J. Zhang, J. Chen, J. Zhu et al. (Tsinghua Univ.) | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | Invariancia exacta por traslación del softmax bajo centrado de claves por media de token $\tilde{K} = K - \bar{k}$: $\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$, eliminando el sesgo de canal antes de los kernels `MXFP4` (`E2M1`) `QK^\top` y `PV` | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA** (*Native Sparse Attention*) y **MoBA** (*Mixture of Block Attention*, Moonshot AI) | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | Atención jerárquica de tres ramas (resúmenes gruesos comprimidos + top-$k$ bloques finos de tokens + ventana deslizante local) y enrutamiento por centroides sin parámetros | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. Factorización espectral con blanqueamiento de entrada, restauración hessiana y MoE de base compartida

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **SVD-LLM** y **SVD-LLM V2** — X. Wang et al. | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | Blanqueamiento de Cholesky $X X^\top = S S^\top$ + SVD de $W S = U \Sigma V^\top$ que proporciona optimalidad exacta de Eckart–Young–Mirsky en norma de salida de activación $\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** y **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | Compensación secuencial de deriva acumulada de activaciones a través de capas truncadas y asignación global de rango según la curvatura de energía singular | `InputWhitenedSVD` (`#552`) |
| **OBR** (*Optimal Brain Restoration*), **MoE-SVD** y **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | Compensación de error hessiano lagrangiana en forma cerrada $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$ que concilia conjuntamente poda + cuantización; base compartida $U_{\text{shared}}$ entre expertos MoE; protección de expertos consciente de modalidad | `OBRCompensator`, `SharedBasisSwitchLinear`, `MAEPAllocator` (`#552`) |

---

## 7. Decodificación especulativa sin pérdidas mediante transporte óptimo y difusión contrastiva

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **SpecTr** (*Optimal Transport Multi-Draft Speculation*) y **Gumbel-Coupled Verification** — Z. Sun, A. T. Suresh et al. | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | Acoplamiento de transporte óptimo sobre $k$ borradores candidatos que maximiza la aceptación preservando estrictamente la distribución objetivo exacta $p_{\text{target}}(x)$ (`TV = 0`) | `SpecTrAndResidualRaceMetal` (`#160`, `#553`) |
| **Residual Race** y **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | Carrera paralela sobre la densidad residual $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ en una sola pasada `simdgroup` sin viajes de ida y vuelta al host | `SpecTrAndResidualRaceMetal` (`#553`) |
| **EAGLE-3** (*Y. Li et al.*), **Token Recycling** (*X. Luo et al.*) y **Load-Aware Dynamic Trees** (*Y. Chung & T. Shin*) | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | Cabeza de borrador de fusión de características multicapa (`low / mid / high`, `fc_fuse`), reciclaje de borradores por adyacencia en búsqueda de prompt y profundidad de árbol adaptativa a la carga y temperatura | `EAGLE3FusionHead`, `TokenRecycling`, `SmallMQuantizedLinear` (`#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | Contraste de logits con decaimiento temporal $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$ que elimina el colapso por repetición en LLM enmascarados / de difusión por bloques | `DiffuCDProcessor` (`#553`) |

---

## 8. Radiografía del modelo, cirugía mecanística, lógica epistémica y sondas autosupervisadas

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **Ley espectral de Marchenko–Pastur**, **geometría de información de Fisher–Rao** y **trirretículo `SIXTEEN_3` de Belnap** — V. A. Marchenko & L. A. Pastur (1967); N. D. Belnap (1977); Y. Shramko & H. Wansing | *Mat. Sb.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | Borde del cuerpo espectral de matrices aleatorias $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$ para diagnóstico de pesos de cola pesada; distancia geodésica esférica $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$; puerta epistémica paraconsistente de cuatro valores verdad/falsedad/conflicto | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **Dirección de característica de rechazo / Abliteración**, **CAA Anti-Steering** (*S. Kitada & S. Hara*, UEC) y **Parcheo de rutas por grupo GQA** (*T. Hara et al.*, Doshisha Univ.) | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Cirugía lineal de dirección en el flujo residual, atribución causal por grupo KV GQA $I(g)$ ($H_g = N_a / N_{kv}$), protección de espacio nulo cognitivo y `MAO-SNMF` (*Lee et al.*, 2026) | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA** (*Y. LeCun, R. Balestriero et al.*) y **Harness²** (*Google Research*) | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | Aprendizaje de representaciones latentes predictivas sin colapso (regularización gaussiana isotrópica `SIGReg`) y orquestación de agentes autoverificadores recursivos | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. Postentrenamiento autorregresivo, optimización distribuida y codiseño para Apple Silicon

| Trabajo | Foro / ID | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **ADPO / cADPO** (*S. Oi et al.*, Science Tokyo / MBZUAI / AIST / NII LLMC), **EMA-QLoRA** (*Gaipnazarov & Zhumakulov*) y **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | Pérdida autorregresiva de preferencias factorizada por prefijo con alineamiento adaptativo de $m$ segmentos (`40.3% -> 45.8%` tasa de victoria LC en AlpacaEval 2), `PHM` hipercomplejo de Kronecker y adaptadores ortogonales `OFT` | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**, **Pipeline sin burbujas (`ZB-H1`)**, **ZeRO-2** y **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | Momento externo de Nesterov federado sobre Thunderbolt 5 / LAN entre varios Mac, división hacia atrás `$B/W$` sin burbujas, estados de optimizador fragmentados y bosquejo de gradientes con realimentación de error | Entrenamiento distribuido y calibración de clúster (`#564`) |
| **Roofline de Apple Silicon, Transformers en ANE y MoE fuera del núcleo** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | Regulación por roofline de ancho de banda de memoria unificada, planificación híbrida GPU + Neural Engine y streaming de expertos desde SSD con `POSIX AIO` sin swap | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. Calibración epistémica europea (FR / DACH), gobernanza, caché semántica y jerarquía UMA (`#621`)

| Trabajo | Foro / Ecosistema | Principio matemático central | Subsistema de Basalt / PRs e Issues |
|---|---|---|---|
| **Score d'influence y enrutamiento calibrado por temperatura** | Calibración ML FR / DACH (2025–2026) | Puntuación de influencia causal por cabeza y puerta de confianza del router calibrada por temperatura | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **QAT STE consciente del dominio y auditoría posterior a la cuantización** | Seguridad y equidad en cuantización (2025–2026) | QAT de dominio mediante estimador directo (straight-through) combinado con auditorías de regresión de calidad, equidad y fugas | `QATFakeQuantizer`, `QuantizationAuditCore` (`#616`, `#623`) |
| **AdaLoRA cuantizado (`LR/40`), SFT→DPO en dos etapas, autoevaluación y auditoría RLVR** | Alineamiento y postentrenamiento (2025–2026) | Tasa de aprendizaje escalada por cuantización (`LR/40`), planificación adaptativa de rango, alineamiento por preferencias en etapas y auditoría de explotación de atajos | `QuantizedLRPolicy`, `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (`#617`, `#624`) |
| **Caché semántica LANTERN y orquestador empresarial de 5 componentes** | Sistemas europeos e IA empresarial (2025–2026) | Reutilización semántica de respuestas indexada por embeddings con obsolescencia acotada y enrutamiento empresarial multicomponente | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **Núcleo de conocimiento neurosimbólico LAAS-CNRS y gobernanza HITL** | LAAS-CNRS / Robótica e IA segura de Toulouse | Restricciones de grafo de conocimiento simbólico emparejadas con puertas deterministas de aprobación Human-in-the-Loop | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **Jerarquía KV escalonada L0/L1/L2 consciente de UMA y planificación heterogénea adaptativa** | Arquitectura de sistemas de memoria unificada (2025–2026) | Presupuestación KV L0/L1/L2 en memoria unificada compartida con promoción `L1 → L0` sin copias y planificación heterogénea de clúster | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
