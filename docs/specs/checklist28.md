# Кластер 28: Обработка ошибок, Retry-логика и отказоустойчивость (Error Handling)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 28.1 | Автоматический Retry | ✅ Done | GemError.swift | Повторный запуск упавших команд с использованием паттернов Exponential Backoff. 30 |
| 28.2 | Предотвращение зацикливания (Loop Prevention) | ✅ Done | GemError.swift | Ограничение количества попыток агента исправить одну и ту же ошибку (Exit Code 53). 10 |
| 28.3 | Классификация ошибок | ✅ Done | GemError.swift | Разделение фатальных ошибок (FatalInputError, FatalSandboxError) от исправимых (Retryable). 10 |
| 28.4 | Анализ стектрейсов моделью | ✅ Done | GemError.swift | Автоматическая передача логов падения stderr обратно в LLM для анализа. 2 |
| 28.5 | Обработка 429 Too Many Requests | ✅ Done | GemError.swift | Умная приостановка выполнения (Sleep) при исчерпании rate limits. 2 |
| 28.6 | Обработка 401 Unauthorized | ✅ Done | GemError.swift | Интеллектуальный перехват истекших токенов OAuth. 3 |
| 28.7 | Обработка 400 Bad Request | ✅ Done | GemError.swift | Специфический парсинг ошибок формата (например, отсутствие reasoning\_content в DeepSeek V4). 3 |
| 28.8 | Валидация вывода (Zod/JSON Schema) | ✅ Done | GemError.swift | Жесткая проверка JSON ответов модели на соответствие контракту (Schema Validation). 1 |
| 28.9 | Защита таблиц форков (Fork Table Protection) | ✅ Done | GemError.swift | Управление процессами в средах с ограниченными ресурсами (WSL2). 65 |
| 28.10 | Fallback на веб\-поиск | ✅ Done | GemError.swift | Переход к Google Search при невозможности самостоятельно разрешить ошибку. 7 |
| 28.11 | Логирование сбоев | ✅ Done | GemError.swift | Сохранение детализированных дампов ошибок в директорию .gemini/logs/ или .claude/. 62 |
| 28.12 | Ручное вмешательство | ✅ Done | GemError.swift | Приостановка зависшей задачи с предложением пользователю помочь агенту. |
