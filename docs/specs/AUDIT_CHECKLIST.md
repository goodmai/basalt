# Чеклист аудита проблем и рисков Gem CLI (mlx)

Этот документ предназначен для регулярной проверки кодовой базы на наличие критических проблем, выявленных в аналогичных системах (Claude Code, Qwen, Gemini).

## 1. Ресурсные риски (Resource Management)
- [ ] **Fork Table Exhaustion**: Проверить `TerminalManager.swift`. Не создаются ли зомби-процессы при частом использовании shell-команд?
- [ ] **Memory Leaks**: Профилировать `RESTServer.swift` и `ModelOrchestratorActor.swift`. Очищается ли память после завершения инференса?
- [ ] **Disk Space**: Проверить `BenchmarkResultStore.swift`. Есть ли лимит на размер хранимых логов и кэшей?

## 2. Безопасность и Изоляция (Security)
- [ ] **Shell Injection**: Проверить все вызовы `Process` и `run_shell_command`. Экранируются ли пользовательские аргументы?
- [ ] **Credential Leaks**: Проверить `CostTracker.swift` и логгеры. Не попадают ли API-ключи в stdout/stderr или файлы логов?
- [ ] **File Locking**: Проверить `MLXInferenceEngine.swift`. Что произойдет, если два агента (или агент и пользователь) одновременно попытаются записать в один файл?
- [ ] **Path Traversal**: Проверить резолвер файлов `@` в `PromptContextBuilder.swift`. Можно ли получить доступ к файлам вне корня проекта (например, `/etc/passwd`)?

## 3. Алгоритмические риски (Agentic Logic)
- [ ] **ReAct Doom Loop**: Убедиться, что в `ModelOrchestratorActor.swift` есть жесткий лимит `maxTurns`. Агент должен остановиться, если не может решить задачу за N шагов.
- [ ] **Context Pollution**: Проверить `TokenBudgetCalculator.swift`. Удаляются ли нерелевантные куски кода из контекста при достижении лимита в 92%?
- [ ] **Forgetfulness Bug**: Проверить логику сжатия в `ContextDegradationProfiler.swift`. Не теряет ли агент важные инструкции при очистке истории?

## 4. UI/UX Стабильность
- [ ] **TTY Detection**: Проверить `TerminalUI.swift`. Корректно ли отображается вывод при перенаправлении в файл (`gem chat > file.txt`)?
- [ ] **ANSI/Escape Sanitization**: Проверить `MarkdownRenderer.swift`. Не ломают ли управляющие символы из кода модели отображение в терминале?
- [ ] **Concurrency Safety**: Проверить `ProgressBar.swift` и `Spinner.swift`. Не возникает ли мерцания при одновременном обновлении нескольких индикаторов?

## 5. Инфраструктурные риски
- [ ] **Auth Refresh Loop**: Проверить `AuthController.swift`. Как система ведет себя при истечении JWT? Нет ли бесконечного цикла запросов?
- [ ] **Rate Limit Fallback**: Проверить `ModelRouter.swift`. Переключается ли система на резервную модель (например, с Qwen на Gemini) при получении HTTP 429?
- [ ] **Unified Memory Pressure**: Проверить `ModelFitAnalyzer.swift`. Учитывается ли давление на память со стороны других приложений macOS?

---
*Дата последнего обновления: 29 апреля 2026 г.*
