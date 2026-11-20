# Basalt für Mac

> Languages: [English](../../mac-app.md) · [Русский](../ru/mac-app.md) · [中文](../zh/mac-app.md) · [Español](../es/mac-app.md) · [Français](../fr/mac-app.md) · [日本語](../ja/mac-app.md) · **Deutsch**

Eine native App rund um Basalt: ein Chat-Fenster und ein Monitor-Fenster auf Basis einer lokalen Engine, die von der App gestartet und überwacht wird. Design und Phasen sind in #434 beschrieben. Dies ist Phase 1.

## Bauen und ausführen

```bash
./scripts/bundle_app.swift            # release build, Metal kernels, then .build/Basalt.app, signed ad hoc
open .build/Basalt.app
```

- `--identity "Developer ID Application: …"` signiert mit Hardened Runtime für die Distribution.
- `--bin PATH` stellt das Bundle aus bereits kompilierten Binaries zusammen.
- `--dry-run` gibt den Ablaufplan aus.

Aus einem Build-Ordner ohne Bundling heraus verweist man die App per Umgebungsvariable auf die Engine: `BASALT_HELPER=.build/release/Basalt .build/release/BasaltMac`.

## Funktionsweise

- **Die Engine ist ein separater Prozess:** das gebündelte `basalt serve` in `Contents/Helpers`.
  - Die App startet sie auf einem freien Loopback-Port (`--port 0 --announce-ready`) und übergibt ihr pro Start ein Token in `BASALT_API_KEY`.
  - Die Kommunikation mit der Engine erfolgt über dieselbe REST-API, die auch Editoren und Skripte nutzen.
- **Abstürze:** Ein Absturz der Engine führt nicht zum Absturz der App. Der Supervisor startet sie nach 1 s, dann nach 2 s usw. bis zu 30 s neu und bricht nach drei Abstürzen innerhalb einer Minute unter Anzeige der letzten Engine-Ausgabe ab.
- **Beenden:** Beim Beenden der App wird die Engine geordnet gestoppt: zuerst SIGINT, nach 10 s SIGKILL.
- **Erneuter Start:** Die App übernimmt eine noch laufende Engine anhand ihrer Protokolldatei (`~/Library/Application Support/Basalt/engine.json`, Berechtigung 0600), sodass das Modell nicht erneut geladen werden muss.
- **Konversationen** werden mit SwiftData in `~/Library/Application Support/Basalt/Conversations.store` gespeichert. Eine durch einen Absturz unterbrochene Antwort wird als gestoppt markiert zurückgegeben.
- **Das Modell** wird in den Einstellungen festgelegt (Standard: `mlx-community/gemma-4-12B-it-qat-4bit`). Die Änderung greift beim nächsten Start der Engine.

## Menüleiste und Quick Ask

- **Die Engine startet mit der App**, nicht erst mit dem Fenster. Das Menüleisten-Symbol (ein Würfel) ist auch dann sichtbar, wenn alle Fenster geschlossen sind. Es zeigt den Zustand sowie das Modell der Engine an und bietet folgende Aktionen:
  - Neue Konversation und Quick Ask;
  - die fünf zuletzt verwendeten Konversationen;
  - Engine starten, neu starten und stoppen;
  - Monitor, Einstellungen und Beenden.
- **Quick Ask:** ⌥Space öffnet aus jeder App heraus ein schwebendes Panel – auch über Vollbild-Apps –, ohne dieser App den Fokus zu entziehen.
  - Frage eingeben und ↩ drücken. Die Antwort wird direkt im Panel gestreamt.
  - ⌘↩ öffnet den Verlauf als reguläre Konversation im Hauptfenster.
  - Esc schließt das Panel und verwirft den Verlauf. Bis ⌘↩ wird nichts aus Quick Ask gespeichert.
  - Das Tastenkürzel wird über `RegisterEventHotKey` registriert und benötigt keine Bedienungshilfen-Berechtigung. Es lässt sich in den Einstellungen deaktivieren; diese Änderung wird beim nächsten Start wirksam.
- **Einstellungen:**
  - Allgemein: Modell, Quick Ask und „Bei Anmeldung öffnen“ (`SMAppService`; funktioniert, sobald die App signiert und im Programme-Ordner abgelegt ist).
  - Cloud-Schlüssel: Umgebungsvariablen wie `OPENROUTER_API_KEY`, deren Werte im Schlüsselbund (Dienst `app.basalt.mac`) gesichert werden. Die Engine erhält sie beim nächsten Start für Compose-Cloud-Modelle. Die App schreibt sie niemals auf den Datenträger.
- **Zustandswiederherstellung:** Die ausgewählte Konversation wird nach einem Neustart wiederhergestellt, ebenso ungesendete Entwürfe pro Konversation aus den User Defaults.

## Modelle

- **Das Modelle-Fenster** (⌥⌘L oder „Modelle“ in der Menüleiste) listet die Modelle im Hugging-Face-Cache (`~/.cache/huggingface/hub`) samt Größe und einer Kompatibilitätsbewertung („fit verdict“) für diesen Mac über `GET /v1/models/preview`.
- **Laden** schaltet die Engine auf das gewählte Modell um (`POST /v1/models/load`) und legt es als Startmodell fest. Ein Modell mit der Bewertung „passt nicht“ kann aus der App heraus nicht geladen werden; das Fenster nennt den Grund.
- **Herunterladen** akzeptiert eine Repo-ID wie `mlx-community/gemma-4-12B-it-qat-4bit`. Die App lädt im eigenen Prozess herunter, zeigt den Fortschritt an und merkt sich unvollständige Downloads. Beim nächsten Start setzt sie diese anhand der Teildateien fort. Ein fehlgeschlagener Download wartet auf einen erneuten Versuch.
- **Löschen** fragt zuerst nach und entfernt dann den Ordner des Modells aus dem Cache. Das aktuell geladene Modell kann nicht gelöscht werden.

## Anhänge

- **Anhängen** über die Büroklammer oder per Drag-and-drop in die Konversation: PDFs, Bilder (PNG, JPEG, HEIC, TIFF, GIF, WebP) und Audio (WAV, MP3, M4A, AAC, FLAC, CAF) mit jeweils bis zu 32 MB.
- **Eine Nachricht mit Anhängen durchläuft `/v1/pipeline`.** Die Engine liest Dokumente und Bilder mit ihrem OCR-Modell oder – falls keines konfiguriert ist – mit Apple Vision. Audio wird mit dem Sprachmodell oder dem Chat-Modell transkribiert. Anschließend beantwortet das Chat-Modell die Nachricht (bzw. „Summarize the attached files.“, wenn die Nachricht leer ist).
- **Der erfasste Text bleibt in der Konversation erhalten.** Jeder Anhang erscheint als Chip, der den von der Engine gelesenen Text offenlegt. Nachfolgende Nachrichten werden mit diesem Text vor der Frage gesendet, sodass sich Folgefragen auf die Datei beziehen können.
- **Der Beleg („Receipt“)** unter der Antwort listet jede Pipeline-Stufe mit Modell und Laufzeit in Millisekunden auf.

## Inspektor

- **⌥⌘I** oder ein Klick auf eine Antwort öffnet den Inspektor neben der Konversation.
- **Konversation:** das Modell sowie ein System-Prompt für diese Konversation. Er wird mit der Konversation gespeichert und jeder Nachricht vorangestellt. Ein leerer Prompt löscht ihn.
- **Antwort:** der Beleg der ausgewählten Antwort: Modell, Prompt- und Completion-Tokens, Time to First Token, Geschwindigkeit sowie jede Pipeline-Stufe mit Modell und Millisekunden.
- **Anhänge:** der Text, den die Engine Seite für Seite aus jeder Datei der Anfrage ausgelesen hat.

## Kurzbefehle, Dienste und Dock

- **Kurzbefehle und Spotlight:** „Ask Basalt“, „Summarize File“, „Transcribe Audio“ und „Switch Model“ sind App Intents. Kurzbefehle führt sie aus, während die App im Hintergrund läuft. Jeder Intent wartet, bis die Engine bereit ist (bis zu drei Minuten), und „Switch Model“ lehnt Modelle ab, die nicht in den Speicher passen. `scripts/bundle_app.swift` extrahiert die Metadaten der Intents mit `appintentsmetadataprocessor` nach `Contents/Resources/Metadata.appintents`, da SwiftPM dies nicht übernimmt.
- **Dienste:** Text in beliebigen Apps markieren und *Dienste → Ask Basalt* wählen. Eine neue Konversation öffnet sich mit diesem Text als Frage.
- **Ein-Klick-Start via `basalt://` von Hugging Face:** `Basalt.app` registriert das URL-Schema `basalt` (`CFBundleURLTypes`). Das Öffnen von `basalt://run?model=<owner/repo>[&role=chat|mtp|moe-stream|ocr|asr|decide|embed|rerank]` – oder ein Klick auf **Run with Basalt** auf einer beliebigen [`goodmai`](https://huggingface.co/goodmai)-Modellkarte – validiert die Hugging-Face-Repo-ID, aktualisiert die gespeicherte Modellauswahl, holt Basalt in den Vordergrund und schaltet die überwachte Engine auf dieses Modell bzw. diese Pipeline-Rolle um.
- **Dock:** Wird eine PDF-, Bild- oder Audiodatei auf das Dock-Symbol gezogen, startet eine neue Konversation mit dieser Datei im Anhang.
- **Mitteilungen:** Wenn eine Antwort fertiggestellt wird, während Basalt im Hintergrund ist, zeigt eine Mitteilung die ersten Zeilen an. Die Berechtigung wird einmalig abgefragt.

## Compose-Systeme und Cloud

- **Einstellungen → Engine:** Entweder ein einzelnes Modell ausführen oder ein Compose-System aus einer Datei (`basalt serve --compose <file>`). „Engine damit neu starten“ beendet die Engine und startet sie mit den neuen Argumenten. Die Auswahl wird auch beim nächsten Start verwendet.
- **Übernahme anhand der Argumente:** Die App übernimmt eine noch laufende Engine nur dann, wenn sie mit denselben Argumenten gestartet wurde. Nach einer Änderung des Modells oder der Compose-Datei startet eine neue Engine. Beim Beenden der App wird auch eine übernommene Engine gestoppt.
- **Monitor:** Zeigt für ein Compose-System Name, Pipelines und die heutigen Cloud-Ausgaben sowie pro Modell Status (`ready`, `loading`, `no-key`, `over-budget` usw.), Keep-Policy, Ausgaben und Obergrenze sowie etwaige Ablehnungen. Cloud-Modelle tragen ein Wolkensymbol.
- **Die Cloud-Markierung:** Eine Antwort, die ein Cloud-Modell genutzt hat, zeigt „☁︎ cloud“ samt Kosten in der Belegzeile und im Inspektor an. Alles ohne diese Markierung lief lokal auf diesem Mac.

## Bedienungshilfen, Sprachen und Updates

- **VoiceOver:**
  - Jedes reine Symbol-Steuerelement besitzt einen gesprochenen Namen: Menüleisten-Symbol, Senden, Stopp, Anhängen, Löschen und Inspektor.
  - Jede Nachricht wird als ein Element vorgelesen: „Du: …“ bzw. „Basalt: …“.
  - Fortschrittsanzeigen benennen, worauf gewartet wird.
  - Das Kompatibilitäts-Badge meldet „passt auf diesen Mac“ oder „passt nicht“.
  - Ausstehende Anhänge werden als „<file> entfernen“ vorgelesen.
- **Sprachen:** Die App folgt der Systemsprache. Derzeit werden Englisch und Russisch ausgeliefert. Eine Übersetzung liegt unter `Localization/<language>.lproj/Localizable.strings`, das `scripts/bundle_app.swift` in das Bundle kopiert und in `CFBundleLocalizations` einträgt. Ein Test schlägt fehl, sobald eine View einen String ergänzt, der in der russischen Tabelle fehlt.
- **Updates:** Einmal täglich fragt die App über die GitHub-API nach der neuesten Basalt-Version. Eine neuere Version (kein Entwurf, keine Vorabversion) erscheint als „Update verfügbar“ in der Menüleiste samt Link zur Release-Seite. Die Anfrage sendet nichts außer der IP-Adresse und dem User-Agent der App an GitHub. Dies lässt sich unter *Einstellungen → Allgemein* deaktivieren.

## Auslieferung: Signiert, notarisiert und als Homebrew-Cask

- **Lokal:** `./scripts/bundle_app.swift --identity "Developer ID Application: Name (TEAMID)" --notarize basalt-notary --dmg` signiert mit Hardened Runtime und erstellt `Basalt-<version>.zip`. Das Archiv wird mit `notarytool` eingereicht, das Ticket an die App geheftet (Stapling), erneut gezippt und schließlich eine `.dmg`-Datei erzeugt.
  - `--zip` allein paketiert ohne Notarisierung.
  - `--notarize` lehnt eine Ad-hoc- oder fehlende Signatur ab.
  - Das Schlüsselbund-Profil wird einmalig mit `xcrun notarytool store-credentials basalt-notary --apple-id <id> --team-id <TEAMID> --password <app-specific password>` angelegt.
- **Bei einem Release:** Sind im Repository die Secrets `MAC_SIGNING_IDENTITY` (Name des Zertifikats im Schlüsselbund des Runners) und `MAC_NOTARY_PROFILE` (das Schlüsselbund-Profil) hinterlegt, baut, signiert und notarisiert ein `v*`-Tag die App automatisch.
  - `Basalt-<version>.zip` wird dem GitHub-Release angehängt.
  - `Casks/basalt-app.rb` wird aus `Casks/basalt-app.rb.template` im selben PR aktualisiert, der auch die Formula anpasst.
  - Ohne diese Secrets veröffentlicht das Release lediglich die Formula mit einem entsprechenden Hinweis.
- **Installation:** Sobald ein Release die App enthält, installiert man sie per `brew install --cask goodmai/basalt/basalt-app`.

## Compose Video Studio, Universal Converter & Live Hub (`v0.12`)

- **Compose Video Studio:** Phasenweise 24-GB-UMA-Videogenerierung und Inline-Wiedergabe (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`) mit `.mp4`-Export (`#611`).
- **Universal Converter (`ConvertModelSheet`):** Vollständige Any-to-Any-Quantisierungs-UI (`MACaQ` Dense/MoE, `FlatQuant`, `LIRQuant`, `SINQ`, `QTIP`, `E8P`, `PV-Tuning`, `QAT STE` + `QuantizationAuditCore`) auf Englisch und Russisch (`#546`, `#599`, `#623`).
- **Live Hub & 10-Hz-SSE-Telemetrie:** Basierend auf `swift-huggingface` mit Xet-Downloads, Live-Hub-Suche/-Veröffentlichung sowie 10-Hz-Streaming-Telemetrie im Monitor (`#587`).

## Budgets

Gemessen auf einem M4 Pro mit 24 GB bei Commit 530aa28 mit Gemma 4 12B in der Engine (#434):

| Budget | Gemessen | Limit | Methode |
|---|---|---|---|
| Fenster beim Warmstart | 257 ms Median aus 5 (Kaltstart: 1.6 s) | 400 ms | `swift scripts/app_budgets.swift` |
| Idle-Wakeups nach Engine-Start | 0 in 10 s | 0 | ebenso |
| App-Prozess | 40 MB, zuzüglich 33.5 MB für ein geöffnetes Transkript mit 2,000 Nachrichten | 150 MB | ebenso und `AppBudgetTests` |
| Gestreamtes Token in ein Transkript mit 2,000 Nachrichten | 7.4 ms p95, 6.0 ms Median | 16.7 ms (60 fps) | `BASALT_APP_BUDGETS=1 swift test --filter AppBudgetTests` |
| Quick Ask: Panel und erstes Token abzüglich Engine-Eigenzeit | 40 ms warm (Panel 15 ms); erste SwiftUI-View im Prozess 175 ms | 150 ms | ebenso |

`app_budgets.swift` belegt das GPU-Lock, da die Engine ihr Modell lädt. Beide Prüfungen sollten auf einem ruhigen Mac ohne laufenden CI-Job ausgeführt werden. Drei Dinge erfordern weiterhin eine manuelle Kontrolle außerhalb der Automatisierung: das Panel über einer Vollbild-App (es ist ein nicht-aktivierendes Panel mit `.fullScreenAuxiliary`), App Intents aus Kurzbefehlen bei geschlossener App sowie eine Cask-Installation auf einem sauberen Mac, wofür die Signatur-Secrets benötigt werden.
