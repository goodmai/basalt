# План разработки Gem CLI (PLAN.md)

**Продукт:** Сервер инференса LLM для Apple Silicon и мультиагентный роутер.
**Текущая версия:** v0.5.0-beta
**Статус:** Активная разработка (Epic 16, 18, 24)

---

## Бэклог предстоящих эпиков

### Эпик 18: Rainbow AI Chat UI (Metal Interface) 🌈
*Детальная спецификация: [docs/EPIC_18_RAINBOW_UI.md](docs/EPIC_18_RAINBOW_UI.md)*
- [ ] **Metal Renderer**: Высокопроизводительный кастомный UI на базе Metal без UIKit.
- [ ] **Rainbow Aesthetics**: Анимированные переливы фона, плейсхолдеров и текста.
- [ ] **Agentic Testing**: Внедрение невидимых маркеров Accessibility для автоматизированного тестирования.
- [ ] **Rich Rendering**: Отрисовка картинок, блоков кода и diff'ов внутри Metal-конвейера.

### Багфикс Эпик: Rainbow UI Input Fixes 🐛
- [x] **Нативный NSTextField**: SwiftUI TextField заменен на `NSViewRepresentable` обертку (`NativeInputField`), что решает проблемы с `makeFirstResponder` и отсутствием поля ввода (field editor) в иерархии AppKit.
- [x] **Фриз первой буквы в интерфейсе чата**: При отправке сообщения (нажатие Enter) строка `@Binding var text` очищалась, но активный `fieldEditor` сохранял "фантомный" введенный символ. Исправлено принудительной очисткой активного `fieldEditor.string` в `updateNSView`.
- [x] **Нет респонса при стриминге (MainThread starvation)**: `MTKView` обновлялся с помощью кастомного GCD-таймера 60 раз в секунду, который полностью блокировал `DispatchQueue.main` и мешал обработке MLX токенов в `MainActor.run`. Исправлено переводом `MTKView` на нативный цикл (CADisplayLink) и явным вызовом `setNeedsDisplay` при обновлении стейта.
- [x] **Зависание процесса (Невозможно завершить задачу Gemm)**: Приложение не закрывалось при закрытии окна крестиком (`cmd+w`). В `RainbowAppDelegate` добавлен метод `applicationShouldTerminateAfterLastWindowClosed`, возвращающий `true`.
- [x] **TDD Покрытие**: Написан набор тестов (`RainbowUIFocusTests` и `RainbowUIGenerationTests`), проверяющих фокус, ввод кириллицы и транзиции состояний генерации.

### Эпик 20: MVI Event-Driven Architecture & UI Bugfixes 🚀
*Детальная спецификация: [docs/EPIC_20_MVI_EVENT_DRIVEN_ARCHITECTURE.md](docs/EPIC_20_MVI_EVENT_DRIVEN_ARCHITECTURE.md)*
- [ ] **MVI Architecture**: Разделение на `LLMEnginePort`, `RenderState`, `TokenEvent`.
- [ ] **Event Bus & Shortcuts**: Перехват и обработка событий ввода (`Ctrl+C`, `Ctrl+D`, `Shift+Tab`, `/btw`).
- [ ] **Throttling & Performance**: Рендер с ограничением 30 FPS без MainThread starvation.
- [ ] **UI Rendering Events**: Корректный перенос введенного сообщения в лог чата и отрисовка картинок/diff'ов/tool calls.

### Эпик: OpenClaw Integration — Мультимодальный Роутер 🦾
*Детальная спецификация: [docs/EPIC_OPENCLAW.md](docs/EPIC_OPENCLAW.md)*
*Чеклист функций: [docs/specs/CHECKLIST_OPENCLAW.md](docs/specs/CHECKLIST_OPENCLAW.md)*
- [ ] **OpenClaw Router**: Команда `gem openclaw` для диспетчеризации агентов.
- [ ] **Audio Pipeline**: Интеграция Whisper (ASR) и TTS для голосового управления.
- [ ] **Vision & OCR**: Обработка изображений и видео через VLM модели.
- [ ] **A2A Protocol**: Механизмы поставки данных внешним агентам.

### Эпик: Emoji as Code & Реактивный фидбек ✨
*Спецификация: [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md)*
- [ ] **Emoji Standard**: Реализация системы визуальных префиксов на основе Gitmoji.
- [ ] **Status Animation**: Анимированные эмодзи для длительных процессов.
- [ ] **Emoji Fallback**: Механизм переключения на текст для не-TTY окружений.

### Эпик: Продвинутые операции с файлами (Clusters 3, 4)
- [ ] **FileEditTool**: Частичная модификация строк без перезаписи файла.
- [ ] **GlobTool**: Расширенный поиск по маскам.
- [ ] **File Locking**: Механизм предотвращения гонок при записи.

### Эпик: Когнитивная память и RAG (Cluster 12)
- [ ] **MEMORY.md**: Механизм "память как подсказка" на уровне проекта.
- [ ] **Vector RAG**: Локальный поиск по кодовой базе.

---

## План релизов

### v0.6.0 (июнь 2026)
- Реализация OpenClaw Router (базовая команда).
- Внедрение Emoji as Code.
- Продвинутые инструменты редактирования файлов.

### v0.7.0 (июль 2026)
- Аудио-пайплайн (Whisper + TTS).
- IDE Bridge для VS Code и Zed.

### v1.0.0 (сентябрь 2026)
- Полная мультимодальность (Vision + Video).
- Безопасность (Sandboxing, Permissions).
- Публикация в Mac App Store.

---

## Метрики успеха
- **Покрытие тестами:** > 95% для всех новых инструментов.
- **Производительность:** TTFT < 100мс для локальных моделей 4B.
- **Безопасность:** 0 критических уязвимостей.

---
*Архив выполненных задач: [PLAN_OLD.md](PLAN_OLD.md)*
