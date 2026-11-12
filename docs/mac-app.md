# Basalt for Mac

A native app around basalt: a chat window and a Monitor window, on a local engine that the app starts and supervises. Design and phases are in #434. This is phase 1.

## Build and run

```bash
./scripts/bundle_app.swift            # release build, Metal kernels, then .build/Basalt.app, signed ad hoc
open .build/Basalt.app
```

- `--identity "Developer ID Application: …"` signs with the hardened runtime for distribution.
- `--bin PATH` assembles from binaries already built.
- `--dry-run` prints the plan.

From a build folder without bundling, point the app at the engine: `BASALT_HELPER=.build/release/Basalt .build/release/BasaltMac`.

## How it works

- **The engine is a separate process:** the bundled `basalt serve`, in `Contents/Helpers`.
  - The app starts it on a free loopback port (`--port 0 --announce-ready`) and passes it a per-launch token in `BASALT_API_KEY`.
  - It reaches the engine over the same REST API that editors and scripts use.
- **Crashes:** a crash of the engine is not a crash of the app. The supervisor restarts it after 1 s, then 2 s and so on up to 30 s, and gives up after three crashes in a minute, showing the engine's last output.
- **Quitting:** quitting the app stops the engine gracefully: SIGINT, then SIGKILL after 10 s.
- **Relaunching:** the app adopts an engine that is still running from its record file (`~/Library/Application Support/Basalt/engine.json`, mode 0600), so the model is not loaded again.
- **Conversations** are stored with SwiftData in `~/Library/Application Support/Basalt/Conversations.store`. An answer interrupted by a crash comes back marked as stopped.
- **The model** is set in Settings (default `mlx-community/gemma-4-12B-it-qat-4bit`). It applies at the engine's next start.

## Menu bar and quick ask
- **The engine starts with the app**, not with the window. The menu bar item (a cube) is there even when every window is closed. It shows the engine's state and model and has these actions:
  - new conversation and quick ask;
  - the five most recent conversations;
  - start, restart and stop the engine;
  - the Monitor, Settings and Quit.
- **Quick ask:** ⌥Space opens a floating panel from any app, including a full-screen one, without taking that app's focus.
  - Type a question and press ↩. The answer streams in the panel.
  - ⌘↩ opens the exchange as an ordinary conversation in the main window.
  - Esc closes the panel and discards the exchange. Until ⌘↩, nothing from quick ask is stored.
  - The shortcut is registered with `RegisterEventHotKey`, which needs no Accessibility permission. It can be turned off in Settings; that change applies at the next launch.
- **Settings:**
  - General: the model, quick ask, and open at login (`SMAppService`; it works once the app is signed and in Applications).
  - Cloud Keys: environment variables, such as `OPENROUTER_API_KEY`, whose values are kept in the Keychain (service `app.basalt.mac`). The engine receives them at its next start, for compose cloud models. The app never writes them to disk.
- **State restoration:** the selected conversation comes back after a relaunch. So does an unsent draft, per conversation, kept in the user defaults.

## Models
- **The Models window** (⌥⌘L, or Models in the menu bar) lists the models in the Hugging Face cache (`~/.cache/huggingface/hub`) with their size and a fit verdict for this Mac, from `GET /v1/models/preview`.
- **Load** switches the engine to that model (`POST /v1/models/load`) and makes it the model the engine starts with. A model whose verdict is "does not fit" cannot be loaded from the app; the window says why.
- **Download** takes a repo id such as `mlx-community/gemma-4-12B-it-qat-4bit`. The app downloads in its own process, shows progress, and remembers unfinished downloads. On the next launch it resumes them from the partial files. A failed download waits for Retry.
- **Delete** asks first, then removes the model's folder from the cache. The loaded model cannot be deleted.

## Attachments
- **Attach** with the paper clip or by dropping files on the conversation: PDFs, images (PNG, JPEG, HEIC, TIFF, GIF, WebP) and audio (WAV, MP3, M4A, AAC, FLAC, CAF), up to 32 MB each.
- **A message with attachments goes through `/v1/pipeline`.** The engine reads documents and images with its OCR model, or Apple Vision without one. It transcribes audio with its speech model, or the chat model. The chat model then answers the message (or "Summarize the attached files." when the message is empty).
- **The perceived text stays in the conversation.** Each attachment shows as a chip, and the chip discloses the text the engine read. Later messages are sent with that text in front of the question, so follow-ups can refer to the file.
- **The receipt** under the answer lists each pipeline stage with its model and milliseconds.

## Inspector
- **⌥⌘I**, or clicking an answer, opens the inspector beside the conversation.
- **Conversation:** the model, and a system prompt for this conversation. It is saved with the conversation and sent before every message. An empty prompt clears it.
- **Answer:** the selected answer's receipt: model, prompt and completion tokens, time to first token, speed, and each pipeline stage with its model and milliseconds.
- **Attachments:** the text the engine read from each file of the question, page by page.

## Shortcuts, Services and the Dock
- **Shortcuts and Spotlight:** Ask Basalt, Summarize File, Transcribe Audio and Switch Model are App Intents. Shortcuts runs them with the app in the background. Each waits until the engine is ready (up to three minutes), and Switch Model refuses a model that does not fit. `scripts/bundle_app.swift` extracts the intents' metadata into `Contents/Resources/Metadata.appintents` with `appintentsmetadataprocessor`, because SwiftPM does not.
- **Services:** select text in any app and choose Services → Ask Basalt. A new conversation opens with that text as the question.
- **Dock:** a PDF, image or audio file dropped on the Dock icon starts a conversation with it attached.
- **Notifications:** when an answer finishes while Basalt is in the background, a notification shows its first lines. The permission is asked once.

## Compose systems and the cloud
- **Settings → Engine:** run one model, or a compose system from a file (`basalt serve --compose <file>`). "Restart the Engine with This" stops the engine and starts it with the new arguments. The choice is also used at the next launch.
- **Adoption is by arguments:** the app adopts an engine left running only if it was started with the same arguments. After a change of model or compose file, a new engine starts. Quitting stops an adopted engine as well.
- **Monitor:** for a compose system, its name, pipelines and cloud spend today, and per model its state (`ready`, `loading`, `no-key`, `over-budget` and so on), keep policy, spend and cap, and any refusal. Cloud models carry a cloud symbol.
- **The cloud mark:** an answer that used a cloud model says "☁︎ cloud" with its cost on the receipt line and in the inspector. Everything without the mark ran on this Mac.

## Accessibility, languages and updates
- **VoiceOver:**
  - Every icon-only control has a spoken name: the menu bar item, send, stop, attach, delete and the inspector.
  - Each message reads as one element, "You: …" or "Basalt: …".
  - Progress indicators say what they wait for.
  - The fit badge says "fits on this Mac" or "does not fit".
  - Pending attachments read as "Remove <file>".
- **Languages:** the app follows the system language. English and Russian ship now. A translation is `Localization/<language>.lproj/Localizable.strings`, which `scripts/bundle_app.swift` copies into the bundle and lists in `CFBundleLocalizations`. A test fails when a view gains a string that the Russian table lacks.
- **Updates:** once a day the app asks the GitHub API for the latest basalt release. A newer, non-draft, non-pre-release version shows as "Update available" in the menu bar, with a link to the release page. The request sends nothing but the app's IP address and user agent to GitHub. Turn it off in Settings → General.

## Shipping: signed, notarized, and a Homebrew cask
- **Locally:** `./scripts/bundle_app.swift --identity "Developer ID Application: Name (TEAMID)" --notarize basalt-notary --dmg` signs with the hardened runtime and writes `Basalt-<version>.zip`. It submits the zip with `notarytool`, staples the ticket to the app and zips it again, then writes a `.dmg`.
  - `--zip` alone packages without notarizing.
  - `--notarize` refuses an ad hoc or missing signature.
  - Create the keychain profile once with `xcrun notarytool store-credentials basalt-notary --apple-id <id> --team-id <TEAMID> --password <app-specific password>`.
- **In a release:** when the repository has the secrets `MAC_SIGNING_IDENTITY` (the certificate's name in the runner's keychain) and `MAC_NOTARY_PROFILE` (that keychain profile), a `v*` tag builds, signs and notarizes the app.
  - It attaches `Basalt-<version>.zip` to the GitHub release.
  - It writes `Casks/basalt-app.rb` from `Casks/basalt-app.rb.template` in the same PR that moves the formula.
  - Without the secrets, the release ships the formula only, with a notice.
- **Install:** once a release carries the app, run `brew install --cask goodmai/basalt/basalt-app`.
