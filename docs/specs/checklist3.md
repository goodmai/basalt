# Кластер 3: Операции чтения файловой системы (FS Read)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 3.1 | Инструмент FileReadTool | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Чтение текстовых файлов с возможностью указания диапазона строк. 2 |
| 3.2 | Экстракция из бинарных форматов | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Распознавание и парсинг текста из PDF-документов и изображений. 2 |
| 3.3 | Инструмент read\_many\_files | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Пакетное асинхронное чтение множества файлов или целых директорий. 19 |
| 3.4 | Валидация ограничений (maxResultSize) | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Ограничение объема считываемых байтов за один вызов. 1 |
| 3.5 | Инструмент GlobTool | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Поиск путей файлов по шаблонам глоббинга (например, \*\*/\*.spec.ts). 2 |
| 3.6 | Парсинг Jupyter Notebooks | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Специализированный парсинг .ipynb с сохранением структуры ячеек. 2 |
| 3.7 | Мониторинг изменений (FS Watch) | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Подписка на системные события изменения файлов. |
| 3.8 | Обработка прав доступа | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Перехват ошибок EACCES / Permission Denied при чтении. 10 |
| 3.9 | Разрешение символических ссылок | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Безопасный переход по симлинкам внутри рабочей директории. |
| 3.10 | Инструмент TaskListTool / /files | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Вывод списка загруженных в контекст файлов или запущенных задач. 2 |
| 3.11 | Хэширование файлов (SHA) | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Сверка хэш-сумм файлов перед чтением. |
| 3.12 | Интеграция с .gitignore | ⚠️ Partial | MLXInferenceEngine.swift, TerminalManager.swift | Игнорирование скрытых файлов и папок node\_modules. |
