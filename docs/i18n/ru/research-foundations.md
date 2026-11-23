# Математические и научные основы Basalt

> Languages: [English](../../research-foundations.md) · **Русский** · [中文](../zh/research-foundations.md) · [Español](../es/research-foundations.md) · [Français](../fr/research-foundations.md) · [日本語](../ja/research-foundations.md) · [Deutsch](../de/research-foundations.md) · [한국어](../ko/research-foundations.md) · [Bahasa Indonesia](../id/research-foundations.md)

Basalt опирается на достижения мирового сообщества открытой науки. Каждое Metal-ядро, кодек квантования, компрессор KV-кэша, спекулятивный верификатор, эпистемический шлюз и целевая функция пост-обучения в Basalt имеют строгое обоснование в рецензируемой математической теории и препринтах открытого доступа из международных научных экосистем — охватывающих **североамериканскую, европейскую, китайскую, российскую, южнокорейскую и японскую** научные школы.

---

## Благодарности и признательность

Мы выражаем искреннюю благодарность и глубокое уважение всем исследователям, математикам, системным инженерам, университетским лабораториям и мейнтейнерам открытого ПО, чьи теоретические прорывы, доказательства и эталонные реализации сделали создание Basalt возможным:

- **Apple Machine Learning Research и команда MLX** (*Awni Hannun, Jagrit Digani, Angelos Katharopoulos, Ronan Collobert* и участники проектов [`ml-explore/mlx`](https://github.com/ml-explore/mlx), [`mlx-swift`](https://github.com/ml-explore/mlx-swift) и [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)) — за вычисления над массивами в унифицированной памяти, примитивы пользовательских ядер Metal и исследования трансформеров для Apple Neural Engine.
- **Лаборатория Cornell University RelaxML** (*Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa*) — за теорию $\mu$-некогерентности, решёточное кодирование `QuIP#` на базе решётки Госсета $E_8$ и решётчато-треллисное квантование со сдвигом битов `QTIP`.
- **Huawei Computing Systems Lab (CSL), Noah's Ark Lab и команда MindIE** (*Lukas Muller, Yuxuan Sun, Ruizhou Ding* и коллеги) — за формулировки `MACaQ`, `SINQ`, `KVarN`, `FlatQuant`, `EAQuant` и микромасштабирование `MXAttention`.
- **IST Austria, Yandex Research, ВШЭ, МФТИ, МГУ и Математический институт им. В. А. Стеклова РАН (МИАН)** (*Dan Alistarh, Elias Frantar, Vage Egiazarian, Денис Кузнеделев, Руслан Свирщевский, Борис Кашин, Иван Оселедец, Д. Р. Пойманов, Д. Н. Давыдова* и коллеги) — за `GPTQ`, `AQLM`, `PV-Tuning`, квантование LLM в переполненных базисах, взвешенные маршрутизатором гессианы MoE и двухуровневый поиск смешанной точности.
- **Университет Цинхуа, DeepSeek-AI, Moonshot AI, Шанхайский университет Цзяо Тун и Li Auto** (*Jintao Zhang, Jianfei Chen, Jun Zhu, Haotian Tang, Zhijian Liu, Song Han, Jingwen Leng, Minyi Guo* и коллеги) — за `SageAttention` (`1 / 2++ / 3`), `Quest`, `DuoAttention`, Native Sparse Attention (`NSA`), `MoBA`, `LIRQuant` и `ParoQuant`.
- **Google Research, Google DeepMind, Йель, Стэнфорд, MIT и Калифорнийский университет в Беркли** (*Amir Zandieh, Insu Han, Praneeth Kacham, Vahab Mirrokni, Ziteng Sun, Ananda Theertha Suresh, Arthur Douillard, Yann LeCun, Randall Balestriero* и коллеги) — за шенноновское KV-квантование по теории скорость-искажение (`QJL`, `PolarQuant`, `TurboQuant`), спекулятивную верификацию нескольких черновиков на основе оптимального транспорта (`SpecTr`), `DiLoCo`, `LeJEPA` и `Harness²`.
- **Японское сообщество NLP и ИИ (`NLP2026`, `JSAI2026`, Science Tokyo, UEC, Ун-т Досися, Токийский ун-т, AIST, NII LLMC, MBZUAI)** (*S. Oi, M. Ukai, M. Kaneko, N. Okazaki, N. Inoue, T. Hara, N. Yoshimaru, K. Hatano, S. Kitada, S. Hara, S. Watanabe, K. Taura* и коллеги) — за авторегрессионную оптимизацию предпочтений (`ADPO` / `cADPO`), причинно-следственный патчинг путей на уровне групп GQA, анализ анти-управления (anti-steering) и остаточное KV-квантование.
- **Южнокорейское сообщество ИИ и систем (`KIISE`, `KICS`, Ун-т Чунан, Нац. ун-т Канвон, Ун-т Ханьян, Сеульский городской ун-т, Nota AI)** (*H. Chung, Y. Chung, T. Shin, A. Lee, H. Hwang, C. Lee, B. Kang, K. Oh, C. Zeng* и коллеги) — за кросс-языковую инвариантность `HeadKV`, разложение полисемантических признаков `MAO-SNMF`, динамические спекулятивные деревья с учётом нагрузки, `FLRot` и каскадно-устойчивое квантование MoE.
- **Открытые сообщества `ggml` / `llama.cpp` и Hugging Face** (*Georgi Gerganov, Iwan Kawrakow*, `mlx-community` и авторы открытых весов по всему миру) — за матрицы значимости `K`-quant / `IQ`, `safetensors`, `swift-transformers` и открытую науку о весах моделей.

---

## 1. Локальные и аффинные преобразования базиса (`W4A4` / `MXFP4` / `W4A16`)

Современное пост-тренировочное квантование подавляет выбросы с тяжёлыми хвостами в весах и активациях ($\max_i |x_i| / \|x\|_2$) с помощью ортогональных преобразований Уолша — Адамара, кронекеровских сопряжений общих линейных групп и разреженных вращений Гивенса, встраиваемых непосредственно в прологи Metal `simdgroup`:

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **MACaQ**: *Model Agnostic Calibrated Quantization* — L. Muller et al. (Huawei CSL) | [arXiv:2611.10649](https://arxiv.org/abs/2611.10649) · [GitHub](https://github.com/huawei-csl/MACaQ) | Строго локальный внутрислойный блочный базис Уолша — Адамара $H_B \in \{H_{32}, H_{64}\}$ + покоординатный спуск по гессиану **Flip-Refinement** в формате OCP `MXFP4` (`E2M1` + `E8M0`) без ограничений на межслойные вращения | `MACaQRefiner`, `MACaQQuantizedLinear`, `MACaQFusedMatmul`, `GGUFMacAqRefiner` (`#535`, `#537`, `#547`, `#549`) |
| **FlatQuant**: *Flatness Matters for LLM Quantization* — Y. Sun et al. (Huawei Noah's Ark / Tsinghua / CUHK) | ICML 2025 · [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | Ослабляет ортогональность Штифеля $Q \in O(d)$ до обратимых кронекеровских множителей $P = P_1 \otimes P_2 \in GL(d_1)\otimes GL(d_2)$, снижая стоимость сопряжения с $O(d^2)$ до $O(d(\sqrt{d_1}+\sqrt{d_2}))$ | `FlatQuantLinear` (`#547`, `#554`) |
| **LIRQuant** и **ParoQuant** — X. Xu et al. (Li Auto); Z. Liu et al. | [arXiv:2610.25675](https://arxiv.org/abs/2610.25675) · [arXiv:2511.10645](https://arxiv.org/abs/2511.10645) | Локальные изометрические / попарные вращения Гивенса $G(i, j, \theta)$, сохраняющие $L_2$-норму в аппаратно-параллельных непересекающихся парах каналов + поканальное масштабирование | `LIRQuantLinear`, `BasaltMetalSubstrate` (`#547`, `#554`) |
| **ThinQuant** и **FLRot** — A. Makni et al.; B. Kang et al. (Hanyang Univ.) | [arXiv:2609.36120](https://arxiv.org/abs/2609.36120) · *KICS Winter 2026* | Структурированные малопараметрические / полностью обучаемые вращения для подавления выбросов в периферийных и низкобитных LLM | `#547`, `#554`, `#563` |

---

## 2. Бистохастическая нормализация Синкхорна и предобусловливание матриц без калибровки

По теореме Синкхорна попеременное масштабирование строк и столбцов приводит неотрицательные модули весов к равномерной дисперсии одновременно вдоль обеих осей, уменьшая шаг квантования в худшем случае без калибровочных данных:

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **SINQ**: *Sinkhorn-Normalized Quantization* — L. Muller et al. (Huawei CSL) | ICML 2026 · [arXiv:2509.22944](https://arxiv.org/abs/2509.22944) · [GitHub](https://github.com/huawei-csl/SINQ) | Двухосевая нормализация Синкхорна — Кноппа $\widetilde{W} = D_1^{-1} W D_2^{-1}$, минимизирующая дисбаланс матрицы $\mathcal{I}(W) = \max_{i,j}|W_{ij}| / \text{median}(\sigma_r, \sigma_c)$ за $O(m n T)$ | `SINQNormalizer`, `WeightConverter+SINQ` (`#548`, `#561`) |
| **No-Overhead SINQ** | [arXiv:2611.05927](https://arxiv.org/abs/2611.05927) | Точное алгебраическое поглощение масштабного множителя столбцов $D_2$ в предшествующий слой `RMSNorm` $\gamma \leftarrow \gamma \odot d_2$ с нулевыми накладными расходами на умножение во время инференса | `SINQAbsorption` (`#548`, `#561`) |
| **A-SINQ** и **EAQuant** (Huawei Noah's Ark) | [arXiv:2512.14531](https://arxiv.org/abs/2512.14531) · [arXiv:2506.13329](https://arxiv.org/abs/2506.13329) | Учитывающее активации целевое взвешивание столбцов $c_j \propto \|X_{:,j}\|_2^\alpha$ + сохранение логитов маршрутизатора и баланса экспертов MoE | `SINQNormalizer`, `SINQAbsorption` (`#548`, `#561`) |

---

## 3. Евклидовы решётки, треллисное кодирование, Бабай — ЛЛЛ и пост-квантовательный `PV-Tuning`

Векторные и треллисные квантователи используют выигрыш от сферической упаковки многомерных решёток ($E_8$, Барнса — Уолла, треллисы на сдвиговых регистрах) и точную оптимизацию ближайшего вектора / KL-дивергенции относительно учителя:

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **QTIP**: *Quantization with Trellises and Incoherence Processing* — A. Tseng et al. (Cornell RelaxML) | NeurIPS 2024 · [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) · [arXiv:2605.10655](https://arxiv.org/abs/2605.10655) | Случайная некогерентность Адамара (`RHT`) + вычисляемый в регистрах битовый треллис со сдвигом (`1MAD` / `3INST`), достигающий искажения вблизи гауссовской границы при `2.0–3.0 bpw` | `QTIPQuantizer`, `QTIPLinear`, `QTIPMetalKernels` (`#409`, `#494`) |
| **QuIP#**: *Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* — A. Tseng et al. | ICML 2024 · [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) · [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) | `E8P` — 2-битный кодбук решётки Госсета $E_8$ ($256$ базисных векторов оболочки чётной чётности $\times 2^8$ знаков $= 65,536$ кодовых слов в $\mathbb{R}^8$, `2.0 bpw`) | `E8PLattice`, `E8PMetalKernels`, `E8PLinear` (`#549`, `#554`) |
| **GPTQ как ближайшая гиперплоскость Бабая**, **MR-GPTQ** и **TurboBoA** | ICLR 2026 · [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) · [arXiv:2602.02545](https://arxiv.org/abs/2602.02545) · [arXiv:2603.02170](https://arxiv.org/abs/2603.02170) | Эквивалентность метода `GPTQ` с обратной связью по ошибке через разложение Холецкого ($H = L L^\top$) алгоритму ближайшей гиперплоскости Бабая для задачи CVP; редукция базиса решётки **Ленстры — Ленстры — Ловаса (`LLL`)** снижает дефект ортогональности; `TurboBoA` на дополнении Шура ускоряет доводку | `BabaiLLLCompensator` (`#549`, `#554`) |
| **PV-Tuning** и **AQLM** — M. Malinovskii, D. Mazur, V. Egiazarian, D. Alistarh et al. | NeurIPS 2024 / ICML 2024 · [arXiv:2405.14852](https://arxiv.org/abs/2405.14852) · [arXiv:2401.06118](https://arxiv.org/abs/2401.06118) | Попеременная сквозная оптимизация по KL-дивергенции учителя для непрерывных масштабов/нулевых точек (`P`) и дискретных назначений (`V`) поверх произвольных квантованных представлений | `PVTuner`, `PVTuner+MACaQ`, `PVTuner+QTIP`, `UniversalModelConverter` (`#410`, `#546`) |
| **Router-Weighted MoE GPTQ** (`Д. Р. Пойманов`, МГУ) и **DARTS-QAT** (`Д. Н. Давыдова`, МГУ) | *Труды ИСП РАН* 38(6), 2026 · *Интеллектуальные системы* 30(3), 2026 | Взвешенный маршрутизатором гессиан MoE $H_i^{\text{RW}} = 2(X_i \odot c_i^\top)X_i^\top$, достигающий `2.8 bpw` (`INT2/INT3`) на `Qwen3-30B-A3B` и `Mixtral-8x7b`; двухуровневая смешанная точность с мёртвой зоной и ограничением по размеру | `GSQRCOConverter`, `#563` |

---

## 4. Теоретико-информационное сжатие KV-кэша и сублинейный поиск по контексту

Чтобы удерживать сессии с длинными цепочками рассуждений (`Long-CoT`) на 32k–131k токенов в оперативной памяти без смещения скалярного произведения, Basalt объединяет 2-битные сетки с балансировкой дисперсии, несмещённые знаковые скетчи Джонсона — Линденштрауса, сферические координаты и калиброванный по точности выбор голов и страниц:

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **KVarN**: *2D Tile Variance Normalization for KV Cache* — Huawei CSL | [arXiv:2606.03458](https://arxiv.org/abs/2606.03458) · [GitHub](https://github.com/huawei-csl/KVarN) | Двухосевая балансировка дисперсии тайлов по каналам и токенам перед сверхнизкобитным KV-квантованием | `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **QJL**, **PolarQuant** и **TurboQuant** — A. Zandieh, I. Han, P. Kacham, V. Mirrokni et al. | AAAI 2025 / AISTATS 2026 / ICLR 2026 · [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) · [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) · [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) | Асимметричная оценка $\widehat{\langle q, k\rangle} = \langle q, \hat{k}_{\text{MSE}}\rangle + \frac{\sqrt{\pi/2}}{m}\|r\|_2 \langle S q, \operatorname{sign}(S r)\rangle$, обеспечивающая **строго нулевое смещение** $\mathbb{E}[\widehat{\langle q, k\rangle}] = \langle q, k\rangle$ вблизи шенноновской границы скорость-искажение; полярно-угловое кодирование векторов ключей | `PolarKeyRows`, `ZonedKVCache+MiddleKeys`, `KVarNQJLCodec`, `KVarNFlashDecodeMetal` (`#550`) |
| **ChanMix**, **ThinKV** и **RQ-KV** (*Watanabe & Taura*, Univ. of Tokyo) | [arXiv:2604.18555](https://arxiv.org/abs/2604.18555) · [arXiv:2510.01290](https://arxiv.org/abs/2510.01290) · *NLP2026* | Смешанная точность для каналов с выбросами (`4-bit` чувствительные / `2-bit` основные), адаптивное сжатие спанов рассуждений (`<think>` против ответа) и центроидно-остаточное KV-квантование | `ChanMixSelector`, `ThinKVSegmenter` (`#550`, `#563`) |
| **Quest**, **DuoAttention**, **RazorAttention**, **HeadKV** и **Cross-Lingual HeadKV** (*H. Chung et al.*, Chung-Ang Univ.) | ICML 2024 / ICLR 2025 / *KoreaScience 2026* · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [arXiv:2407.15891](https://arxiv.org/abs/2407.15891) · [arXiv:2410.19258](https://arxiv.org/abs/2410.19258) | Внимание top-$k$ по ограничивающим прямоугольникам страниц (`Quest`), разделение голов на поисковые и потоковые по выходному KL и кросс-языковая инвариантность поисковых голов ($\text{RBO} \approx 0.8$) | `QuestKVCache`, `HeadSplitKVCache`, `KVHeadSelection` (`#37`, `#282`, `#563`) |
| **CacheBlend** — J. Yao et al. | EuroSys 2025 · [arXiv:2405.16444](https://arxiv.org/abs/2405.16444) · [arXiv:2410.15332](https://arxiv.org/abs/2410.15332) | Выборочный перерасчёт KV на головах внимания с наибольшим дрейфом для композиции непрефиксных фрагментов | `CacheBlend` (`#36`, `#83`, `#164`, `#454`) |

---

## 5. Аппаратно-выровненное микромасштабирование и обучаемое блочно-разреженное внимание

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **MXAttention**: *Microscaling Quantized Attention* — Huawei MindIE Team | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) | **Единое онлайн-перемасштабирование (`UOS`)**, объединяющее обновления максимума онлайн-softmax $m_i$ со степенными сдвигами показателя `E8M0` + **нормализация с сохранением разбиения единицы (`PNQ`)**, гарантирующая $\sum_j \tilde{P}_{ij} = 1$ для квантованных вероятностей | `SageMXAttentionMetal` (`#551`) |
| **SageAttention**, **SageAttention2++** и **SageAttention3** — J. Zhang, J. Chen, J. Zhu et al. (Tsinghua Univ.) | ICLR 2025 / ICML 2025 / NeurIPS 2025 · [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) · [arXiv:2505.21136](https://arxiv.org/abs/2505.21136) · [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | Точная трансляционная инвариантность softmax при центрировании ключей по среднему токенов $\tilde{K} = K - \bar{k}$: $\operatorname{softmax}(Q K^\top / \sqrt{d}) \equiv \operatorname{softmax}(Q \tilde{K}^\top / \sqrt{d})$, устраняющая смещение по каналам перед `MXFP4` (`E2M1`) ядрами `QK^\top` и `PV` | `SageMXAttentionMetal` (`#551`) |
| **DeepSeek NSA** (*Native Sparse Attention*) и **MoBA** (*Mixture of Block Attention*, Moonshot AI) | ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | Трёхветвевое иерархическое внимание (сжатые грубые сводки + top-$k$ детальных блоков токенов + локальное скользящее окно) и беспараметрическая центроидная маршрутизация | `BlockSparseRoutingMetal` (`#551`) |

---

## 6. Спектральная факторизация с отбеливанием входов, восстановление гессианом и MoE с общим базисом

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **SVD-LLM** и **SVD-LLM V2** — X. Wang et al. | ICLR 2025 · [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) · [arXiv:2503.12340](https://arxiv.org/abs/2503.12340) | Отбеливание Холецкого $X X^\top = S S^\top$ + SVD матрицы $W S = U \Sigma V^\top$ даёт точную оптимальность Эккарта — Янга — Мирского в норме выходов активаций $\|X W^\top - X W_r^\top\|_F^2 = \sum_{i > r} \sigma_i^2$ | `InputWhitenedSVD` (`#552`) |
| **SAES-SVD** и **HiSVD** | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) · ICLR 2026 | Последовательная компенсация накопленного дрейфа активаций по усечённым слоям и глобальное распределение рангов по кривизне сингулярной энергии | `InputWhitenedSVD` (`#552`) |
| **OBR** (*Optimal Brain Restoration*), **MoE-SVD** и **MAEP** | [arXiv:2606.09080](https://arxiv.org/abs/2606.09080) · ICLR 2026 | Замкнутая лагранжева компенсация ошибки по гессиану $\Delta W^* = -E H^{-1} C^\top (C H^{-1} C^\top)^{-1}$, совместно согласующая прунинг и квантование; общий базис $U_{\text{shared}}$ между экспертами MoE; модально-ориентированная защита экспертов | `OBRCompensator`, `SharedBasisSwitchLinear`, `MAEPAllocator` (`#552`) |

---

## 7. Беспотерное спекулятивное декодирование на оптимальном транспорте и контрастная диффузия

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **SpecTr** (*Optimal Transport Multi-Draft Speculation*) и **Gumbel-Coupled Verification** — Z. Sun, A. T. Suresh et al. | NeurIPS 2023 / 2024 · [arXiv:2310.15141](https://arxiv.org/abs/2310.15141) · [arXiv:2408.07978](https://arxiv.org/abs/2408.07978) | Связывание по оптимальному транспорту по $k$ черновикам-кандидатам, максимизирующее принятие при строгом сохранении точного целевого распределения $p_{\text{target}}(x)$ (`TV = 0`) | `SpecTrAndResidualRaceMetal` (`#160`, `#553`) |
| **Residual Race** и **UniVer** | [arXiv:2611.00506](https://arxiv.org/abs/2611.00506) · ICLR 2026 | Параллельная гонка по остаточной плотности $r(v) = \max(0, p(v) - q_{\text{acc}}(v))$ за один проход `simdgroup` без обращений к хосту | `SpecTrAndResidualRaceMetal` (`#553`) |
| **EAGLE-3** (*Y. Li et al.*), **Token Recycling** (*X. Luo et al.*) и **Load-Aware Dynamic Trees** (*Y. Chung & T. Shin*) | NeurIPS 2025 / ACL / *J. KIISE* 53(10), 2026 · [arXiv:2503.01840](https://arxiv.org/abs/2503.01840) · [arXiv:2408.08696](https://arxiv.org/abs/2408.08696) | Черновая голова со слиянием признаков нескольких уровней (`low / mid / high`, `fc_fuse`), переиспользование черновиков по смежности из промпта и глубина дерева, адаптирующаяся к температуре и нагрузке | `EAGLE3FusionHead`, `TokenRecycling`, `SmallMQuantizedLinear` (`#173`, `#208`, `#553`, `#563`) |
| **DiffuCD**: *Contrastive Decoding for Diffusion LLMs* | [arXiv:2610.40613](https://arxiv.org/abs/2610.40613) | Затухающий по шагам времени контраст логитов $\ell_t = (1 + \alpha(t))\ell_{\text{cond}} - \alpha(t)\ell_{\text{amateur}}$, устраняющий коллапс повторений в блочно-диффузионных и маскированных LLM | `DiffuCDProcessor` (`#553`) |

---

## 8. Рентген модели, механистическая хирургия, эпистемическая логика и самообучающиеся зонды

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **Спектральный закон Марченко — Пастура**, **Информационная геометрия Фишера — Рао** и **Трирешётка Белнапа `SIXTEEN_3`** — В. А. Марченко и Л. А. Пастур (1967); N. D. Belnap (1977); Я. Шрамко и H. Wansing | *Мат. сб.* 72(114) · *Modern Uses of Multiple-Valued Logic* · *J. Philos. Logic* | Граница основного спектра случайной матрицы $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$ для диагностики тяжёлых хвостов весов; геодезическое сферическое расстояние $d_{\text{FR}}(p, q) = 2\arccos(\sum_i \sqrt{p_i q_i})$; четырёхзначный паранепротиворечивый эпистемический шлюз «истина / ложь / конфликт» | `ModelXRay`, `EpistemicGate`, `PhasorProbeEvaluator` (`#450`, `#498`) |
| **Направление признака отказа / Аблитерация**, **CAA Anti-Steering** (*S. Kitada & S. Hara*, UEC) и **GQA Group Path Patching** (*T. Hara et al.*, Doshisha Univ.) | NeurIPS 2024 / *NLP2026 B1-13 & B3-11* · [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) · [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) | Линейная хирургия направлений в остаточном потоке, причинная атрибуция по KV-группам GQA $I(g)$ ($H_g = N_a / N_{kv}$), защита когнитивного нуль-пространства и `MAO-SNMF` (*Lee et al.*, 2026) | `MLXInferenceEngine+LabSteer`, `ModelSurgery` (`#44`, `#563`) |
| **LeJEPA** (*Y. LeCun, R. Balestriero et al.*) и **Harness²** (*Google Research*) | [arXiv:2610.14224](https://arxiv.org/abs/2610.14224) · [GitHub](https://github.com/google-research/harness2) | Свободное от коллапса обучение латентных предиктивных представлений (изотропная гауссовская регуляризация `SIGReg`) и рекурсивная самопроверяющая обвязка агентов | `LeJEPATrain`, `BasaltHarness` (`#90`, `#465`) |

---

## 9. Авторегрессионное пост-обучение, распределённая оптимизация и совместное проектирование под Apple Silicon

| Работа | Площадка / ID | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **ADPO / cADPO** (*S. Oi et al.*, Science Tokyo / MBZUAI / AIST / NII LLMC), **EMA-QLoRA** (*Gaipnazarov & Zhumakulov*) и **PHM / OFT** | *NLP2026 P9-7* · *VAK 2026* · NeurIPS / ICLR | Факторизованная по префиксу авторегрессионная функция потерь предпочтений с адаптивным $m$-сегментным выравниванием (`40.3% -> 45.8%` AlpacaEval 2 LC Win Rate), гиперкомплексные кронекеровские адаптеры `PHM` и ортогональные адаптеры `OFT` | `MLXInferenceEngine+LabTrain`, `CorpusTrain`, `DistillTrain` (`#563`) |
| **DiLoCo**, **Zero-Bubble Pipeline (`ZB-H1`)**, **ZeRO-2** и **PowerSGD** | [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) · [arXiv:2401.10241](https://arxiv.org/abs/2401.10241) · [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · [arXiv:1905.13727](https://arxiv.org/abs/1905.13727) | Федеративный внешний импульс Нестерова по Thunderbolt 5 / LAN между несколькими Mac, беспузырьковое расщепление обратного прохода `$B/W$`, шардирование состояний оптимизатора и сжатие градиентов с обратной связью по ошибке | Распределённое обучение и калибровка кластера (`#564`) |
| **Модель пропускной способности Apple Silicon, ANE Transformers и Out-of-Core MoE** | [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) · [Apple MLX M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [Apple MLX Distributed](https://machinelearning.apple.com/research/mlx-distributed-inference) · [arXiv:2402.16363](https://arxiv.org/abs/2402.16363) · [arXiv:2608.16157](https://arxiv.org/abs/2608.16157) | Регулирование темпа по roofline-модели пропускной способности унифицированной памяти, гибридное планирование GPU + Neural Engine и потоковая подкачка экспертов с SSD через `POSIX AIO` без свопинга | `MemoryCoordinator`, `SystemProfiler`, `ComposeProfileCatalog`, `POSIXAIOStreamer` (`#45`, `#225`, `#272`, `#274`, `#399`) |

---

## 10. Европейская (FR / DACH) эпистемическая калибровка, управление, семантическое кэширование и иерархия UMA (`#621`)

| Работа | Площадка / экосистема | Основной математический принцип | Подсистема Basalt / PR и Issue |
|---|---|---|---|
| **Score d'influence и маршрутизация с температурным масштабированием** | Калибровка ML во Франции и регионе DACH (2025–2026) | Поголовная оценка причинного влияния и шлюз уверенности маршрутизатора с температурной калибровкой | `HeadInfluenceCore`, `CalibratedRouterGate` (`#615`, `#622`) |
| **Предметно-ориентированное QAT STE и аудит после квантования** | Безопасность и справедливость квантования (2025–2026) | Предметное QAT с прямым пробросом градиента (straight-through estimator) в паре с регрессионными аудитами качества, справедливости и утечек | `QATFakeQuantizer`, `QuantizationAuditCore` (`#616`, `#623`) |
| **Квантованная AdaLoRA (`LR/40`), двухэтапный SFT→DPO, самооценка и аудит RLVR** | Выравнивание и пост-обучение (2025–2026) | Скорость обучения, масштабированная под квантование (`LR/40`), адаптивное планирование ранга, двухэтапное выравнивание предпочтений и аудит эксплуатации шорткатов | `QuantizedLRPolicy`, `AdaLoRARankScheduler`, `TwoStageSFTDPOPolicy`, `SelfAppraisalCore`, `ShortcutAuditCore` (`#617`, `#624`) |
| **Семантическое кэширование LANTERN и 5-компонентный бизнес-оркестратор** | Европейские системы и корпоративный ИИ (2025–2026) | Семантическое переиспользование ответов с индексацией по эмбеддингам и ограниченным устареванием в сочетании с многокомпонентной корпоративной маршрутизацией | `SemanticCacheCore`, `SemanticCacheActor`, `BusinessOrchestratorCore` (`#618`, `#625`) |
| **Нейро-символическое ядро знаний LAAS-CNRS и управление HITL** | LAAS-CNRS / робототехника и безопасный ИИ в Тулузе | Ограничения символического графа знаний в паре с детерминированными шлюзами одобрения с участием человека (Human-in-the-Loop) | `SymbolicKnowledgeGraph`, `HITLGovernanceCore`, `HITLApprovalActor` (`#619`, `#626`) |
| **UMA-ориентированная трёхуровневая иерархия KV L0/L1/L2 и адаптивное гетерогенное планирование** | Архитектура систем с унифицированной памятью (2025–2026) | Бюджетирование KV по уровням L0/L1/L2 в общей унифицированной памяти с повышением `L1 → L0` без копирования и гетерогенным планированием кластера | `TieredKVHierarchyCore`, `AdaptiveHeterogeneousPlanner` (`#620`, `#627`) |
