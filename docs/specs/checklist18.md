# Кластер 18: Синтетический вывод и форматирование (Formatting Output)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 18.1 | Инструмент SyntheticOutputTool | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Принудительная генерация строго структурированных данных (JSON/XML). 2 |
| 18.2 | Форматирование вывода (/output-style) | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Выбор форматов представления (таблицы, списки, plain text). 2 |
| 18.3 | Управление темами (/theme, /color) | ✅ Done | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Переключение визуальных стилей и отключение ANSI-цветов терминала. 2 |
| 18.4 | Пайплайн результатов (Piping) | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Поддержка передачи вывода утилиты в стандартный поток ОС (\` |
| 18.5 | Экспорт в HTML | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Генерация форматированных HTML-страниц с результатами работы агента. 3 |
| 18.6 | Адаптивный рендеринг таблиц | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Автоматическое сжатие столбцов Markdown-таблиц под ширину окна (TTY). |
| 18.7 | Экранирование спецсимволов | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Безопасный вывод потенциально опасных управляющих символов оболочки. |
| 18.8 | Отключение Markdown (Raw Mode) | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Вывод чистого текста без форматирования звездочками и бэктиками. |
| 18.9 | Управление многословием (/brief) | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Переключение агента в режим лаконичных ответов (Ralph Wiggum Technique). 2 |
| 18.10 | Мульти-проходная генерация (/passes) | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Итеративное улучшение вывода (Multi-pass execution). 2 |
| 18.11 | Локализация UI сообщения | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Перевод системных сообщений CLI (ошибки, прогресс) на разные языки. |
| 18.12 | Стриминг JSON-партов | ❌ Planned | TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift | Потоковая отдача кусков JSON по мере их готовности сервером. 57 |
