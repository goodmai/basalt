# パイプラインプリセットとメモリモード

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · [中文](../zh/pipelines.md) · [Español](../es/pipelines.md) · [Français](../fr/pipelines.md) · **日本語** · [Deutsch](../de/pipelines.md) · [한국어](../ko/pipelines.md) · [Bahasa Indonesia](../id/pipelines.md)

`/v1/pipeline`（MCP `basalt_pipeline`）は、知覚（OCR、音声認識）→ 判断（System One）→ 回答（チャットモデル）を 1 つのプロセス内で実行します。プリセットはあらかじめ用意されたプロファイルの serve フラグを自動的に設定しますが、明示的に指定したフラグが常に優先されます：

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One by readout of the chat model, the chat model
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe and the chat model
basalt serve --pipeline-preset support-triage  # System One with the ticket questions, and the chat model
```

チャットモデルは Gemma 4 12B が使用されます。他のモデルと並べてメモリ上に収まり、さらに 8 GB の余裕がある場合は Qwen3.8-27B 3-bit が使用されます。`--pipeline-memory` はモデル間でメモリを共有する方法を指定します：

- `resident` はすべてのモデルをロードしたまま保持するため、全モデルの合計サイズがワーキングセットに収まる必要があります。
- `sequential` は各ステージの知覚モデル（perceiver）をそのステージ実行時のみロードし、完了後にアンロードします。チャットモデルは最大の知覚モデルと並べてメモリに収まる場合は常駐し、収まらない場合はそのステージの前後でアンロードされます。最大の単一ステージと 2 GB のみがメモリに収まればよいため、メモリ容量の小さい Mac でも重い構成を実行できます（処理時間は長くなります）。sequential モードのパイプラインリクエストは一度に 1 件ずつ処理され、通常のチャットリクエストはモデルの入れ替え中待機します。
- `auto`（デフォルト）は、合計がメモリに収まる場合は resident を選択し、そうでない場合は sequential を選択します。単体でメモリに収まらない大きすぎるステージがある場合のみ拒否します。

M4 Pro、24 GB（ワーキングセット 19.1 GB）、リリースビルド、コミット 1921c3c での実測値です。2 回目のリクエストはページキャッシュがウォームな状態です：

| プリセット | モデル | 実行プラン | リクエスト 1 | リクエスト 2 | うちスワップ時間 | ピークメモリ使用量 |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential（resident には 25.9 GB 必要）、音声処理の前後でチャットモデルをアンロード | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`、チャットモデルは常駐 | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
