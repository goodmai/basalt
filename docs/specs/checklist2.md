# Кластер 2: Управление контекстным окном (Context Window Mgt)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 2.1 | Локальная токенизация | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Подсчет токенов на стороне клиента перед отправкой запроса к LLM. 2 |
| 2.2 | Динамическое сжатие (/compact) | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Удаление промежуточных рассуждений из истории сессии. 2 |
| 2.3 | Инъекция системных промптов | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Слияние пользовательских инструкций с системными ограничениями агента. 16 |
| 2.4 | Token/Prompt Caching | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Оптимизация затрат путем кэширования неизменяемого контекста на стороне API. 7 |
| 2.5 | Визуализация контекста (/context) | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Рендеринг текущего объема памяти и загруженных файлов в UI. 2 |
| 2.6 | Динамический резолвер файлов (@) | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Оператор для быстрого внедрения содержимого файла прямо в строку ввода. 4 |
| 2.7 | Буферизация стриминга | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Агрегация потокового вывода от LLM для устранения мерцания в терминале. 2 |
| 2.8 | Лимитирование ходов (Turn Limits) | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Отслеживание количества итераций в цикле ReAct (Exit Code 53). 10 |
| 2.9 | Обфускация данных (Undercover Mode) | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Удаление специфичных терминов, Slack-каналов и названий компаний. 8 |
| 2.10 | Внедрение Diff-изменений | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Добавление в контекст только измененных строк вместо всего файла. |
| 2.11 | Редактирование секретов среды | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Подмена паролей и токенов в логах перед отправкой в облачную LLM. 11 |
| 2.12 | Каппирование дерева рендеринга | ✅ Done | TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift | Ограничение глубины дерева сообщений в UI для долгих сессий. 3 |
