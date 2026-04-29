# Кластер 1: Инициализация и настройка среды (Environment Init)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 1.1 | Команда /init | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Генерация файлов локального контекста (CLAUDE.md, QWEN.md). 2 |
| 1.2 | Парсинг иерархии settings.json | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Каскадное чтение конфигураций: system \-\> user \-\> project. 10 |
| 1.3 | Интеграция с Keychain/Credential Manager | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Безопасное извлечение API-ключей без их хранения в plain text. 1 |
| 1.4 | Parallel Prefetching | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Асинхронное выполнение сетевых handshake'ов при запуске. 2 |
| 1.5 | Диагностика среды (/doctor) | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Проверка версий Bun/Node, доступности портов и переменных окружения. 2 |
| 1.6 | Валидация стейта через Zod | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Проверка целостности кэшированных файлов перед загрузкой в RAM. 1 |
| 1.7 | Парсинг флагов запуска (Commander.js) | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Обработка CLI-аргументов для переопределения режимов запуска. 1 |
| 1.8 | Инициализация Sandbox | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Подготовка контейнеров (Docker/Podman/Seatbelt) для выполнения кода. 10 |
| 1.9 | Динамическая подгрузка телеметрии | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Ленивый импорт тяжелых модулей OpenTelemetry (opt-out). 1 |
| 1.10 | Детектирование операционной системы | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Адаптация системных вызовов под darwin, win32, linux. 13 |
| 1.11 | Загрузка A/B тестов (Feature Flags) | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Подключение к серверам Statsig для извлечения локального кэша конфигураций. 14 |
| 1.12 | Управление рабочими директориями | ✅ Done | OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift | Установка root-контекста на основе .git или package.json. |
