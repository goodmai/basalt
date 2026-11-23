# 機能とベンチマーク

> Languages: [English](../../benchmarks.md) · [Русский](../ru/benchmarks.md) · [中文](../zh/benchmarks.md) · [Español](../es/benchmarks.md) · [Français](../fr/benchmarks.md) · **日本語** · [Deutsch](../de/benchmarks.md) · [한국어](../ko/benchmarks.md) · [Bahasa Indonesia](../id/benchmarks.md)

## 機能一覧

高速化機能はすべてオプトインであり、行に明記されていない限り greedy 出力は完全に同一のまま維持されます。数値は電源接続した M4 Pro 24 GB、release ビルドでの計測結果です。

| 機能 | フラグ | 内容 | 実測値 |
|---|---|---|---|
| プロンプトルックアップ（投機的デコーディング） | `serve --speculative-lookup` | プロンプトまたは出力からコピーした最大 16 トークンを下書きし、1 回のフォワードパスで検証 | コード編集で 1.84–2.66×（Bonsai 2 27B）、3.45–3.62×（Gemma 4 12B）。散文では約 1.0× |
| MTP ヘッド・ドラフティング | `serve --mtp [--mtp-block N]` | モデルのマルチトークン予測ヘッド（`mtp.safetensors`、MTPLX パックなど）で下書き。任意の温度で厳密一致。ラウンドは下書き 1 トークンから開始し、4 回連続で全採択されると最大 N−1（デフォルト 4）まで増加、棄却されると 1 に戻る | コードで 1.30×、散文で 1.04×、推論で 1.26×（Bonsai 2 27B、T=0.7）。後続ターンは通常デコーディングと同様にプロンプトキャッシュを再利用: 1,149 トークンのターンで TTFT 12.5 s → 0.75 s（Bonsai 2 27B） |
| Ternary Bonsai 2 | （自動） | Hadamard 回転済み三値チェックポイント（`prism_hadamard_qwen35`）を読み込み | 27B モデルを 7.4 GB に収容 |
| 埋め込みとリランク | `serve --embedding-model`, `--rerank-model` | チャットモデルと並行して OpenAI `/v1/embeddings`、Cohere/Jina `/v1/rerank` を提供 | bge-small 229–299 texts/s、Qwen3-Embedding-0.6B 20 texts/s（バッチ 32） |
| ディスク上のプロンプトキャッシュ | `serve --prompt-cache-disk-mb N` | 再開したセッションや再起動したサーバーが、プリフィルし直す代わりに最長一致プレフィックスを復元 | — |
| KV キャッシュ量子化とゾーン | `serve --kv-bits 4\|8\|auto\|k8v4`, `--kv-zones` | 同じメモリ容量に、より長いコンテキストを収容 | 16k プロンプトのピーク 1.03 GB 対 3.11 GB（Qwen3-4B、4-bit ゾーン） |
| 同時実行リクエスト | `serve --max-concurrent N` | 実行中の各リクエストが KV 空き容量を予約し、収まらない場合はメモリ不足で落とさず 503 `Retry-After` を返却 | — |
| DRY および XTC サンプラー（実験的） | `serve --experimental-samplers` | `dry_*` / `xtc_*` リクエストフィールド | ループ防止ツールではない: 2-bit モデルでは DRY が事実誤認の裏にループを隠してしまう（#177） |

## README の数値の出典

README の最初の画面に掲載されている数値はすべて記録済みの実行結果に基づいており、その機能を導入した PR に数値表、コミット、ログパスが記載されています。PR 内に特記がない限り、環境は M4 Pro 24 GB、release ビルドです。

| 主張 | 出典 |
|---|---|
| Qwen3-4B 78–89 tok/s、Ternary Bonsai 2 27B が 7.4 GB で 20.9–21.5 tok/s、コードでのプロンプトルックアップ時 31–38 | [#239](https://github.com/goodmai/basalt/pull/239)（[#218](https://github.com/goodmai/basalt/pull/218), [#231](https://github.com/goodmai/basalt/pull/231), [#232](https://github.com/goodmai/basalt/pull/232) より） |
| Bonsai 2 27B + MTP ヘッド、コードで 27.1 tok/s | [#263](https://github.com/goodmai/basalt/pull/263) |
| Gemma 4 12B でプロンプトルックアップ 3.45–3.62×、greedy 出力は不変 | [#265](https://github.com/goodmai/basalt/pull/265) |
| Ornith-1.5-9B（12.6k トークンプロンプト）で 2 ターン目が 36 秒ではなく約 1 秒 | [#82](https://github.com/goodmai/basalt/pull/82) |
| Qwen3-4B で KV ゾーン使用時に 16k プロンプトのピークが 3.11 GB から 1.03 GB に削減 | [#159](https://github.com/goodmai/basalt/pull/159) |
| モデル混合ルーティング: 全ラベル正解、1000 リクエストあたり $0.035、p95 4.8 s 対 10.0 s | [#418](https://github.com/goodmai/basalt/pull/418) |
| Mac アプリ: ウォームウィンドウ 257 ms、アイドルウェイクアップ 0 回、ストリーミングトークンあたり p95 7.4 ms、クイック質問 40 ms | [#463](https://github.com/goodmai/basalt/pull/463) |
| 24 GB 上で Gemma 4 12B の隣に 9B 判定器を配置: 4-bit が収容可能（15.3 GB）、LLM トークン 77% 削減で精度 98.9% | [#465](https://github.com/goodmai/basalt/pull/465) |
| Qwen3.6-35B-A3B-8bit（37.7 GB）を 24 GB 環境で SSD ストリーミング実行、スワップ 0 で 8.77 tok/s | [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478) |
| 量子化後 `--experimental-pv-tune`: 3 分未満で 2–3-bit のパープレキシティ劣化を 8.7–29% 回復 | [#486](https://github.com/goodmai/basalt/pull/486) |
| QTIP ビットシフト・トレリス `--experimental-qtip`: `Qwen3.8-27B` 2.5 bpw（10.00 GB）が 3-bit affine および PV-Tuning を上回る（16k PPL 4.621 対 4.702 / 4.652）。2.0 bpw（8.48 GB）は 2-bit PPL を 72.1% 削減（5.345 対 19.146） | [#494](https://github.com/goodmai/basalt/pull/494) |
| Metal 上でのネイティブ `.gguf` GSQ-RCO / `i-quant` 実行（`IQ4_XS` / `IQ3_S`） | [#433](https://github.com/goodmai/basalt/pull/433) |
| M4 Pro 24 GB 上で Recursive `Flint-40B-A2.5B`（`oQ3e`）が 17.61 GB で 67.02 tok/s デコード（273.5 tok/s プリフィル） | [#522](https://github.com/goodmai/basalt/pull/522) |
| `Zyphra/ZONOS3-Mini-hf` 44.1 kHz ニューラル TTS が 2.24 GB で RTF 0.228–0.263（EN/DE/RU で実時間の 3.8–4.4×） | [#527](https://github.com/goodmai/basalt/pull/527) |
| 84 種のハードウェア階層別 Compose プロファイル（`16–256 GB × 12`）と `Basalt.app` RAM 階層自動推奨 | [#528](https://github.com/goodmai/basalt/pull/528) |
| シングルパス `MACaQSwitchLinear` WHT + `gatherQuantizedMM`: MoE カーネル `3.41×` 高速化（`2.06 → 0.60 ms`）、Gemma 4 26B-A4B エンドツーエンド `+82%`（`49.2–54.8 tok/s`）、Ornith 35B-A3B 3-bit で `64–75 tok/s` | [#628](https://github.com/goodmai/basalt/pull/628) |
| `Qwen3.5-9B-PV-MACaQ-4bit`: 8-bit と完全一致の `4/5` Ladder、デコード `+52%`（p50 `43.2 tok/s`）、RAM `−33%`（`6.68 vs 9.97 GB`）、最大層ドリフト `3.37%` | [#545](https://github.com/goodmai/basalt/pull/545), [#630](https://github.com/goodmai/basalt/pull/630) |
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`: 24 GB Mac で `15.1 GB` RAM（ピーク `18.09 GB`）、`4/5` Ladder + `4/6` Lunar を p50 `14.1 tok/s` で達成（4-bit 参照は OOM） | [#628](https://github.com/goodmai/basalt/pull/628), [#630](https://github.com/goodmai/basalt/pull/630) |

README に記載された Golden テスト済みモデルは、CI 上で `scripts/check_readme_models.swift` によって検証されます。記載されるモデルは、`Tests/BasaltTests/Fixtures/golden` 内に実在するフィクスチャへリンクしている必要があります。

自己完結型の静的ベンチマークページ（[`docs/benchmark-page/index.html`](../../benchmark-page/index.html)）とレビュー可能な Hugging Face コレクションマニフェスト（[`docs/benchmark-page/hf-collection.json`](../../benchmark-page/hf-collection.json)）は、これらの表、`benchmarks/`、および Golden フィクスチャから `swift scripts/build_benchmark_page.swift` によって生成されます（CI では `--check` で検証され、ローカルで `--logs logs` を渡すとランナーの `summary` 表が付加されます）。外部公開の最終判断は常に人間が行います。

## ベンチマーク

2 種類のハーネスを用意しています。どちらも稼働中のサーバーに対して実行し、ご自身のマシンで再現可能な数値を報告します。

**月へのラダー（The ladder to the moon）** — `benchmarks/ladder/run.py`。下段より厳密に難度が上がる 5 段構成で、最終段は地球から月への完全なミッションです。回答の文章の見た目では一切採点されません。プロンプトに提示されていない入力に対してモデル自身が書いたコードを実際に実行するため、もっともらしく見える誤答はすべて不合格になります。

| 段 | タスク | 実際にテストする内容 |
|---|---|---|
| 1 | 複二次方程式、実数解 | 閉形式解とエッジケース: 解なし、重解、ゼロ |
| 2 | 複二次方程式、複素数解 | 分岐処理 — 重複度を含めた 4 つすべての根 |
| 3 | フーリエ係数 | 数値積分。のこぎり波、矩形波、余弦波で検証 |
| 4 | ポンプ揚程 | 単位換算（mm 対 m、L/s 対 m³/s）+ 閉形式解を持たない Colebrook の式 |
| 5 | **地球 → 月ミッション** | `sqrt(2ηQ)` からの排気速度、各段のツィオルコフスキーの式、4 体軌道伝播、月面制動、軟着陸 — すべての定数はプロンプト内にあり、教科書からの暗記では解けない設計 |

最上段は単一の数値ではなく 6 つの不変条件で採点されるため、エネルギー計算は正しくても軌道伝播を実装していないモデルにもスコアが表示されます。プロンプトは `benchmarks/ladder/lunar_prompt.md` にあり、`lunar_task.md` の参照値はモデルには決して見せません。

`--warmup` を指定すると、インストラクトモデルなら通過できるはずの軽量スモークテスト 3 問（代数、フィボナッチ、翻訳）が先頭に追加されます。これらはラダースコアには含まれません。

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`、公式 pass@2 ルール。コーパスはリポジトリに同梱されていないため、まず `benchmarks/arc-agi/dataset/` に取得してください（`benchmarks/arc-agi/ARC_AGI.md` を参照）。

### 結果

24 GB ユニファイドメモリ搭載の M シリーズ Mac で計測。`thinking` はチャットテンプレートの `<think>` ブロックを指します。推論モデルではデフォルトで有効になっており、`--reasoning-effort none` で無効化できます。

| モデル | ウォームアップ | ラダー | 最上段 | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`（thinking オフ、24 GB） | 3/3 | **4/5** | 4/6 不変条件 | — | 11.6–13.5（p50 14.1） |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`（thinking オフ、24 GB） | 3/3 | **4/5** | 0/6（length） | — | 9.4–11.4（p50 11.6） |
| `Qwen3.5-9B-PV-MACaQ-4bit`（thinking オフ） | 3/3 | **4/5** | 0/6（prose） | — | 35.6–41.6（p50 43.2） |
| `mlx-community/Qwen3.5-9B-8bit`（参照、thinking オフ） | 3/3 | **4/5** | 0/6（prose） | — | 24.7–27.3 |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit`（thinking オフ） | 3/3 | **4/5** | 5/6 不変条件 | — | 56.0–63.4 |
| `goodmai/gemma-4-12B-it-qat-4bit`（thinking オフ） | 3/3 | **4/5** | **6/6 PASS** | — | 18.8–22.2 |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX`（thinking オフ、`#628`） | 3/3 | 3/5 | 0/6 | — | 49.2–54.8 |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`（thinking オフ、`#628`） | 3/3 | 1/5 | 0/6 | — | 64.0–74.7 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit`（thinking オフ） | 3/3 | 1/4 ² | 未実行 | 0/3 ¹ | 約 45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit`（thinking オン） | 2/3 | 0/4 ² | 未実行 | 未実行 | 約 43 |

Ornith 9B の各段ごとの内訳（thinking オフ）:

| 段 | 結果 | 詳細 |
|---|---|---|
| 1 · 複二次方程式（実数解） | ✅ | 重解および解なしケースを含め根が正確 |
| 2 · 複二次方程式（複素数解） | ❌ | `x⁴ + 1` で 4 つではなく 2 つの根しか返さない |
| 3 · フーリエ係数 | ❌ | `f(x) = x` で `a0 = 10.86`（正解は 0） — 式ではなく積分処理の誤り |
| 4 · ポンプ揚程 | ❌ | `v = 0.0298` m/s（正解は 1.79） — mm から m への換算が欠落 |
| 5 · 月ミッション | — | 未実行 |

ウォームアップ（thinking オフ）: 代数 ✅（163 tok、4 s）、フィボナッチ ✅（757 tok、16 s）、翻訳 ✅（303 tok、7 s — ドイツ語は自然な表現、フランス語は直訳ではなく意訳）。

² 段 1 と段 2 は、ラダーが別々のプロンプトに分割される前に計測されたものです。1 つのプロンプトで両方の関数を要求し、実数解の半分は正解、複素数解の半分は不正解でした。段 5 はこのモデルに対してまだ実行されていません。

¹ ARC-AGI 列は部分実行の結果です。学習タスク 3 件（`007bbfb7`、`00d62c1b`、`017c7c7b`）で pass@2 による正解は 0 件、合計 330 秒でした。残り 2 件の前に中断したため、正式スコアではなくスモークテストとして扱ってください。

検証スクリプト自体も、モデルや GPU なしで動作チェックが行われています:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

thinking **オン**の場合、3 つの高難度タスクと翻訳はいずれも `finish_reason: length` で終了しました — トークン予算が `<think>` ブロックに消費され、回答が一切出力されませんでした。これは能力の上限ではなくトークン予算の枯渇による失敗であるため、上の表では両モードを分けて報告しています。

### 再現手順

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

表にモデルを追加するには、それぞれを 1 回ずつ実行するだけです — 新しい行を追加する PR を歓迎します。
