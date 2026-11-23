# Gemini Decision 1 を使ったクラウド判断

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · [中文](../zh/cloud-decisions.md) · [Español](../es/cloud-decisions.md) · [Français](../fr/cloud-decisions.md) · **日本語** · [Deutsch](../de/cloud-decisions.md) · [한국어](../ko/cloud-decisions.md) · [Bahasa Indonesia](../id/cloud-decisions.md)

Google の `gemini-decision-1` は、テキスト、画像、音声、動画、PDF を対象に 1 パスで System One の質問（`noul`、`choice`、`score`）に回答します。ワイヤーフォーマットは basalt 自身の `/v1/systemone` と一致しているため、ローカル判断モデルの前段に置くクラウドステージとして機能します：

```bash
export GEMINI_API_KEY=...        # read from the environment, never a flag or a file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- 確信度が低い質問（`--decision-escalate-below` 未満）、呼び出し失敗、レート制限時はローカルモデルへフォールバックします。
- `/v1/systemone` は `state` の隣に `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]` を受け付けます。メディアを読み取れるステージが存在しない場合、メディアは拒否されます。
- compose ファイルでは、モデルは `source: cloud:gemini/gemini-decision-1` とし、`api_key_env: GEMINI_API_KEY` を指定します。
  - Gemini モデルは判断（decide）カスケード専用として機能します。
  - パイプラインでは、そのモデルに `data: media` が設定されていない限り、知覚（perceive）されたテキストはこの Mac 上に留まります。設定されている場合は、添付ファイル本体と知覚されたテキストの両方が判断のために Google へ送信されます。
- コストは API が報告するトークン使用量から 100 万入力トークンあたり $0.10 で計算され、compose の予算（budget）に計上されます。

## キャリブレーション教師モデルとして

Gemini Decision 1 は単なるラベルではなく、キャリブレーション済みの確率を返します。Google によれば確率キャリブレーションにおいて最高水準の性能を持ちます。このため、basalt のローカル判断モデルにとって理想的なキャリブレーション教師となります。同じ状態を Gemini Decision 1 とローカルの readout の両方に通し、その出力確率に対してローカルの temperature をフィットさせるか、OpenJev ヘッドを学習させます。これにより、ローカルの判断モデルが出力する確信度がクラウド側と同じ意味を持つようになり、エスカレーションの閾値もそのまま引き継ぐことができます。これをキャリブレーションステップへ組み込む作業は、#423 のローカル JEV 対応（J6: バンドルからのキャリブレーション読み込み）とあわせて計画されています。
