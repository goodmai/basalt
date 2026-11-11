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
