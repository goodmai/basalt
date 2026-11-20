# Mac 版 Basalt

> Languages: [English](../../mac-app.md) · [Русский](../ru/mac-app.md) · [中文](../zh/mac-app.md) · [Español](../es/mac-app.md) · [Français](../fr/mac-app.md) · **日本語** · [Deutsch](../de/mac-app.md)

Basalt 向けのネイティブアプリです。チャットウィンドウとモニター（Monitor）ウィンドウを備え、アプリ自身が起動・監視するローカルエンジン上で動作します。設計とフェーズ計画は #434 に記載されています。本ドキュメントはフェーズ 1 の内容です。

## ビルドと実行

```bash
./scripts/bundle_app.swift            # release build, Metal kernels, then .build/Basalt.app, signed ad hoc
open .build/Basalt.app
```

- `--identity "Developer ID Application: …"` は、配布用に Hardened Runtime を有効にして署名します。
- `--bin PATH` は、すでにビルド済みのバイナリからアプリを組み立てます。
- `--dry-run` は実行計画を表示します。

バンドルを作成せずにビルドフォルダから起動する場合は、`BASALT_HELPER=.build/release/Basalt .build/release/BasaltMac` のようにアプリへエンジンのパスを指定してください。

## 仕組み

- **エンジンは別プロセスで動作します:** バンドル内の `Contents/Helpers` にある `basalt serve` が起動します。
  - アプリは空いているループバックポートでエンジンを起動し（`--port 0 --announce-ready`）、起動ごとのトークンを `BASALT_API_KEY` で渡します。
  - アプリは、エディタやスクリプトが利用するのと同じ REST API 経由でエンジンと通信します。
- **クラッシュ時の挙動:** エンジンがクラッシュしてもアプリ自体はクラッシュしません。スーパーバイザーが 1 秒後、2 秒後、と最大 30 秒まで待機時間を延ばしながら再起動を試み、1 分間に 3 回クラッシュした場合は再起動を中止して、エンジンの最後の出力を表示します。
- **終了時の挙動:** アプリを終了すると、エンジンを安全に停止します（SIGINT を送信し、10 秒後に SIGKILL を送信します）。
- **再起動時の挙動:** アプリは記録ファイル（`~/Library/Application Support/Basalt/engine.json`、パーミッション 0600）を参照し、すでに稼働中のエンジンがあれば引き継ぐため、モデルが再読み込みされることはありません。
- **会話履歴**は SwiftData を用いて `~/Library/Application Support/Basalt/Conversations.store` に保存されます。クラッシュにより中断された回答には、停止済み（stopped）のマークが付きます。
- **モデル**は設定（Settings）で指定します（デフォルトは `mlx-community/gemma-4-12B-it-qat-4bit`）。変更は次回のエンジン起動時に適用されます。

## メニューバーとクイック質問

- **エンジンはウィンドウではなくアプリと一緒に起動します。** すべてのウィンドウを閉じていても、メニューバーアイテム（立方体アイコン）は常駐します。ここにはエンジンの状態とモデル名が表示され、以下の操作を実行できます:
  - 新規会話とクイック質問（quick ask）
  - 直近 5 件の会話
  - エンジンの起動（start）、再起動（restart）、停止（stop）
  - モニター（Monitor）、設定（Settings）、終了（Quit）
- **クイック質問（Quick ask）:** ⌥Space を押すと、フルスクリーンアプリを含むどのアプリの上からでも、そのアプリのフォーカスを奪わずにフローティングパネルが開きます。
  - 質問を入力して ↩ を押すと、パネル内に回答がストリーミング表示されます。
  - ⌘↩ を押すと、そのやり取りをメインウィンドウの通常の会話として開きます。
  - Esc を押すとパネルが閉じ、やり取りは破棄されます。⌘↩ を押すまで、クイック質問の内容は一切保存されません。
  - このショートカットは `RegisterEventHotKey` で登録されているため、アクセシビリティ権限は不要です。設定（Settings）で無効化でき、その変更は次回起動時に適用されます。
- **設定（Settings）:**
  - General（一般）: モデル、クイック質問、ログイン時に開く（`SMAppService`、アプリが署名されて「アプリケーション」フォルダに配置されている場合に動作します）。
  - Cloud Keys（クラウドキー）: `OPENROUTER_API_KEY` などの環境変数。値はキーチェーン（サービス名 `app.basalt.mac`）に保持されます。コンポーズ（compose）のクラウドモデル用として、次回のエンジン起動時に渡されます。アプリがこれらのキーをディスクに書き込むことはありません。
- **状態の復元:** 選択していた会話はアプリの再起動後も復元されます。未送信の下書きも会話ごとにユーザーデフォルトに保持され、復元されます。

## モデル

- **モデル（Models）ウィンドウ**（⌥⌘L、またはメニューバーの Models）には、Hugging Face キャッシュ（`~/.cache/huggingface/hub`）内のモデル一覧と、そのサイズ、および `GET /v1/models/preview` から取得したこの Mac での適合判定（fit verdict）が表示されます。
- **Load（読み込み）** はエンジンをそのモデルに切り替え（`POST /v1/models/load`）、次回のエンジン起動時のモデルとしても設定します。判定が「does not fit（この Mac には収まりません）」のモデルはアプリから読み込むことができず、ウィンドウにその理由が表示されます。
- **Download（ダウンロード）** には `mlx-community/gemma-4-12B-it-qat-4bit` などのリポジトリ ID を指定します。アプリは独自プロセスでダウンロードを実行し、進捗を表示し、未完了のダウンロードを記録します。次回起動時には部分ファイルから再開します。失敗したダウンロードは「Retry（再試行）」待ちになります。
- **Delete（削除）** は確認ダイアログのあと、キャッシュからモデルのフォルダを削除します。現在読み込まれているモデルは削除できません。

## 添付ファイル

- ペーパークリップアイコンをクリックするか、会話画面にファイルをドロップして**添付**できます。対応形式は PDF、画像（PNG、JPEG、HEIC、TIFF、GIF、WebP）、音声（WAV、MP3、M4A、AAC、FLAC、CAF）で、各ファイル最大 32 MB までです。
- **添付ファイル付きのメッセージは `/v1/pipeline` を経由します。** エンジンはドキュメントや画像を OCR モデル（未設定の場合は Apple Vision）で読み取ります。音声は音声認識モデル（未設定の場合はチャットモデル）で文字起こしします。その後、チャットモデルがメッセージに回答します（メッセージが空の場合は `"Summarize the attached files."` として処理されます）。
- **読み取られたテキストは会話内に保持されます。** 各添付ファイルはチップとして表示され、チップを展開するとエンジンが読み取ったテキストを確認できます。以降のメッセージではそのテキストが質問の前に付加されて送信されるため、フォローアップの質問でファイルの内容を参照できます。
- 回答の下にある**レシート（receipt）**には、各パイプラインステージと使用モデル、処理時間（ミリ秒）が一覧表示されます。

## インスペクタ

- **⌥⌘I** を押すか回答をクリックすると、会話の横にインスペクタが開きます。
- **Conversation（会話）:** モデルと、この会話専用のシステムプロンプトを設定できます。システムプロンプトは会話とともに保存され、毎回のメッセージの前に送信されます。空にするとクリアされます。
- **Answer（回答）:** 選択した回答のレシート情報（モデル、プロンプトおよび生成トークン数、初回トークンまでの時間、生成速度、各パイプラインステージの使用モデルとミリ秒単位の処理時間）を表示します。
- **Attachments（添付ファイル）:** 質問の各ファイルからエンジンが読み取ったテキストをページごとに表示します。

## ショートカット、サービス、Dock

- **ショートカットと Spotlight:** Ask Basalt、Summarize File、Transcribe Audio、Switch Model は App Intents として実装されています。「ショートカット」アプリからバックグラウンドで実行でき、それぞれエンジンの準備が整うまで待機します（最大 3 分）。Switch Model はメモリに収まらないモデルへの切り替えを拒否します。SwiftPM はインテントのメタデータを抽出しないため、`scripts/bundle_app.swift` が `appintentsmetadataprocessor` を用いて `Contents/Resources/Metadata.appintents` にメタデータを抽出します。
- **サービス（Services）:** 任意のアプリでテキストを選択し、「Services → Ask Basalt」を選ぶと、そのテキストを質問とした新しい会話が開きます。
- **Hugging Face からのワンクリック `basalt://` 起動:** `Basalt.app` は `basalt` URL スキーム（`CFBundleURLTypes`）を登録しています。`basalt://run?model=<owner/repo>[&role=chat|mtp|moe-stream|ocr|asr|decide|embed|rerank]` を開くか、任意の [`goodmai`](https://huggingface.co/goodmai) モデルカードの **Run with Basalt** をクリックすると、Hugging Face リポジトリ ID が検証され、保存されているモデル選択が更新され、Basalt が前面に表示されて監視下のエンジンがそのモデルまたはパイプラインロールに切り替わります。
- **Dock:** PDF、画像、音声ファイルを Dock アイコンにドロップすると、そのファイルを添付した状態で新しい会話が始まります。
- **通知:** Basalt がバックグラウンドにある間に回答が完了すると、最初の数行を含む通知が表示されます。通知の許可は初回のみ求められます。

## コンポーズシステムとクラウド

- **Settings → Engine:** 単一モデルの実行、またはファイルからのコンポーズシステムの実行（`basalt serve --compose <file>`）を選択します。「Restart the Engine with This」をクリックするとエンジンを停止し、新しい引数で再起動します。この選択は次回起動時にも使用されます。
- **引き継ぎは引数の一致で判定されます:** アプリは、同じ引数で起動された場合にのみ稼働中のエンジンを引き継ぎます。モデルやコンポーズファイルを変更した場合は新しいエンジンが起動します。アプリを終了すると、引き継いだエンジンも同様に停止します。
- **Monitor（モニター）:** コンポーズシステムについて、システム名、パイプライン、本日のクラウド利用額、および各モデルの状態（`ready`、`loading`、`no-key`、`over-budget` など）、保持ポリシー（keep policy）、利用額と上限、拒否理由を表示します。クラウドモデルにはクラウドシンボルが付きます。
- **クラウドマーク:** クラウドモデルを使用した回答には、レシート行とインスペクタに「☁︎ cloud」とそのコストが表示されます。このマークがない処理はすべてこの Mac 上で実行されたものです。

## アクセシビリティ、言語、アップデート

- **VoiceOver:**
  - アイコンのみのすべてのコントロール（メニューバーアイテム、送信、停止、添付、削除、インスペクタ）に読み上げ用の名前が設定されています。
  - 各メッセージは「You: …」または「Basalt: …」の 1 つの要素として読み上げられます。
  - プログレスインジケータは何を待機しているかを読み上げます。
  - 適合バッジは「fits on this Mac」または「does not fit」と読み上げられます。
  - 送信待ちの添付ファイルは「Remove <file>」として読み上げられます。
- **言語:** アプリはシステム言語に従います。現在は英語とロシア語が同梱されています。翻訳ファイルは `Localization/<language>.lproj/Localizable.strings` に配置され、`scripts/bundle_app.swift` がバンドルにコピーして `CFBundleLocalizations` に登録します。ビューに新しい文字列が追加され、ロシア語テーブルに不足がある場合はテストが失敗します。
- **アップデート:** アプリは 1 日 1 回 GitHub API に最新の basalt リリースを問い合わせます。ドラフトでもプレリリースでもない新しいバージョンがある場合、リリースページへのリンクとともにメニューバーに「Update available」と表示されます。このリクエストではアプリの IP アドレスとユーザーエージェント以外の情報は GitHub に送信されません。「Settings → General」から無効にできます。

## 配布: 署名、公証、Homebrew Cask

- **ローカルでのビルド:** `./scripts/bundle_app.swift --identity "Developer ID Application: Name (TEAMID)" --notarize basalt-notary --dmg` を実行すると、Hardened Runtime で署名して `Basalt-<version>.zip` を作成します。その後、`notarytool` で zip を提出し、アプリにチケットをステープルして再圧縮し、`.dmg` を生成します。
  - `--zip` のみを指定すると、公証なしでパッケージ化します。
  - `--notarize` はアドホック署名または署名なしの状態を拒否します。
  - キーチェーンプロファイルは事前に 1 回だけ `xcrun notarytool store-credentials basalt-notary --apple-id <id> --team-id <TEAMID> --password <app-specific password>` で作成しておきます。
- **リリース時:** リポジトリにシークレット `MAC_SIGNING_IDENTITY`（ランナーのキーチェーン内の証明書名）と `MAC_NOTARY_PROFILE`（上記のキーチェーンプロファイル）が設定されている場合、`v*` タグのプッシュでアプリのビルド、署名、公証が自動実行されます。
  - GitHub リリースに `Basalt-<version>.zip` が添付されます。
  - formula を更新する同じ PR 内で、`Casks/basalt-app.rb.template` から `Casks/basalt-app.rb` が生成されます。
  - シークレットが設定されていない場合、リリースには注意書きとともに formula のみが公開されます。
- **インストール:** リリースにアプリが含まれた後は、`brew install --cask goodmai/basalt/basalt-app` を実行してインストールできます。

## コンポーズ動画スタジオ、ユニバーサルコンバーター & ライブ Hub (`v0.12`)

- **Compose Video Studio:** 段階的 24 GB UMA 動画生成とインライン再生（`LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video`、`Wan 2.2`）および `.mp4` エクスポート（`#611`）。
- **ユニバーサルコンバーター (`ConvertModelSheet`):** 英語・ロシア語対応の完全な any-to-any 量子化 UI（`MACaQ` dense/MoE、`FlatQuant`、`LIRQuant`、`SINQ`、`QTIP`、`E8P`、`PV-Tuning`、`QAT STE` + `QuantizationAuditCore`）（`#546`、`#599`、`#623`）。
- **ライブ Hub & 10 Hz SSE テレメトリ:** `swift-huggingface` を基盤とした Xet ダウンロード、ライブ Hub 検索・公開、Monitor での 10 Hz ストリーミングテレメトリ（`#587`）。

## パフォーマンスバジェット

`530aa28` 時点の M4 Pro（24 GB）上で、エンジンに Gemma 4 12B を読み込んで計測した値です（#434）:

| バジェット | 実測値 | 上限 | 計測方法 |
|---|---|---|---|
| ウォーム起動時のウィンドウ表示 | 中央値 257 ms（5 回計測、コールド起動: 1.6 s） | 400 ms | `swift scripts/app_budgets.swift` |
| エンジン起動後のアイドル時ウェイクアップ | 10 秒間で 0 回 | 0 | 同上 |
| アプリプロセス | 40 MB（2,000 メッセージの履歴を開いた状態で +33.5 MB） | 150 MB | 同上、および `AppBudgetTests` |
| 2,000 メッセージの履歴へのストリーミングトークン描画 | p95 で 7.4 ms、中央値 6.0 ms | 16.7 ms (60 fps) | `BASALT_APP_BUDGETS=1 swift test --filter AppBudgetTests` |
| クイック質問: エンジン処理時間を除くパネル表示と初回トークン | ウォーム時 40 ms（パネル表示 15 ms）、プロセス内最初の SwiftUI ビューは 175 ms | 150 ms | 同上 |

エンジンがモデルを読み込むため、`app_budgets.swift` は GPU ロックを取得します。CI ジョブが動いていない静かな Mac 上で両方実行してください。なお、自動化されていない手動確認項目として、フルスクリーンアプリ上でのパネル表示（`.fullScreenAuxiliary` を指定した非アクティブ化パネル）、アプリを閉じた状態での「ショートカット」からの App Intents 実行、およびクリーンな Mac での Cask インストール（署名シークレットが必要）があります。
