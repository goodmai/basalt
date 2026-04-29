# Кластер 26: Управление LLM-моделями и генерацией (Model Switching)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 26.1 | Переключение моделей (/model) | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Горячая смена активной LLM (Opus, Sonnet, Gemini Pro, DeepSeek V4). 2 |
| 26.2 | Регулировка креативности (/effort) | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Управление системными параметрами температуры (temperature) и top\_p. 2 |
| 26.3 | Локальные модели (Ollama) | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Интеграция с локальными серверами LLM (Ollama/Llama.cpp). 3 |
| 26.4 | Мультиплексирование провайдеров | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Поддержка OpenAI, Anthropic, Gemini-compatible API одновременно. 5 |
| 26.5 | Управление квантованием | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Выбор размера весов (4-bit, 8-bit) для локальных моделей. |
| 26.6 | Fallback-маршрутизация | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Автопереключение на резервную модель при падении основной (HTTP 500/429). |
| 26.7 | Контроль контекстного окна | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Динамическая адаптация размера входных данных под лимиты модели (1M токенов у Gemini). 7 |
| 26.8 | Тюнинг параметров (Parameters overrides) | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Тонкая настройка штрафов (frequency\_penalty, presence\_penalty). 39 |
| 26.9 | Выбор "Thinking Models" | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Активация моделей с расширенной фазой рассуждений (Reasoning content). 3 |
| 26.10 | Оптимизация латенси (Fast Mode) | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Команда /fast для переключения на сверхбыстрые модели (Haiku/Flash). 2 |
| 26.11 | Балансировка нагрузки | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Распределение запросов между несколькими API ключами. |
| 26.12 | Управление кэшированием (Prompt Caching logic) | ✅ Done | ModelRouter.swift, ModelsCommand.swift | Форсирование структуры промпта для максимального попадания в кэш. 7 |
