# Кластер 6: Среда исполнения (Shell Execution)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 6.1 | Инструмент BashTool | ✅ Done | TerminalManager.swift | Выполнение произвольных команд в bash-окружении (shell:true). 2 |
| 6.2 | Инструмент PowerShellTool | ✅ Done | TerminalManager.swift | Нативная поддержка выполнения скриптов для сред ОС Windows. 2 |
| 6.3 | Инструмент REPLTool | ✅ Done | TerminalManager.swift | Запуск интерактивной песочницы (Python/Node) для вычисления логики. 2 |
| 6.4 | Инструмент run\_shell\_command (Qwen) | ✅ Done | TerminalManager.swift | Обертка с обязательным запросом подтверждения от пользователя. 19 |
| 6.5 | Перехват потоков (stdout/stderr) | ✅ Done | TerminalManager.swift | Стриминг консольного вывода команды напрямую в контекст агента. 19 |
| 6.6 | Таймауты и прерывания процессов | ✅ Done | TerminalManager.swift | Принудительное завершение команд (SIGKILL), зависших дольше лимита. |
| 6.7 | Санитаризация входных данных | ✅ Done | TerminalManager.swift | Экранирование переменных для предотвращения Shell Injection. 22 |
| 6.8 | Поддержка псевдотерминалов (PTY) | ✅ Done | TerminalManager.swift | Корректный парсинг команд, требующих интерактивного ввода. |
| 6.9 | Фоновое исполнение (nohup) | ✅ Done | TerminalManager.swift | Запуск демонизированных процессов без блокировки CLI. |
| 6.10 | Обработка Exit Codes | ✅ Done | TerminalManager.swift | Использование кодов возврата для ветвления логики агента. 10 |
| 6.11 | Ограничение ресурсов (cgroups) | ✅ Done | TerminalManager.swift | Лимитирование потребления CPU/RAM дочерними процессами. |
| 6.12 | Фильтрация ANSI Escape-кодов | ✅ Done | TerminalManager.swift | Очистка вывода от служебных символов раскраски текста. |
