# Кластер 20: Диагностика, дебаггинг и мониторинг (Diagnostics)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 20.1 | Дебаггинг вызовов (/debug-tool-call) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Детальная инспекция конкретного вызова функции (Tool Call) с входными/выходными JSON параметрами. 2 |
| 20.2 | Визуализация графа (/ctx\_viz) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Отрисовка структуры текущего контекстного окна, памяти и загруженных модулей. 2 |
| 20.3 | Трассировка API (/ant-trace) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Внутренняя система логирования всех HTTP-запросов к провайдерам (Anthropic/Google). 2 |
| 20.4 | Дамп памяти (/heapdump) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Экспорт состояния оперативной памяти процесса Bun/NodeJS. 2 |
| 20.5 | Инвалидация кэшей (/break-cache) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Принудительная очистка всех локальных кэшей токенов, файлов и конфигураций. 2 |
| 20.6 | Сброс лимитов (/reset-limits) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Обнуление счетчиков локальных rate-limit'ов для целей тестирования. 2 |
| 20.7 | Профилирование (/perf-issue) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Сбор и отправка отчета о производительности утилиты (Flame Graphs). 2 |
| 20.8 | Отправка багрепортов (/bug / /feedback) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Интегрированная система сбора диагностической информации и отправки её разработчикам. 2 |
| 20.9 | Транспортные логи MCP | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Просмотр сырых JSON-RPC и SSE сообщений между агентом и серверами. 29 |
| 20.10 | Вывод стектрейсов ошибок | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Парсинг и форматирование логов падения внутренних процессов утилиты. |
| 20.11 | Мокирование лимитов (/mock-limits) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Искусственное воспроизведение ситуации исчерпания API бюджетов. 2 |
| 20.12 | Воспроизведение размышлений (/thinkback) | ✅ Done | SystemProfiler.swift, ModelFitAnalyzer.swift | Анимированный повторный показ процесса "мышления" LLM. 2 |
