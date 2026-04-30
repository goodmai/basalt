# Кластер 19: Аутентификация, сессии и состояние (Auth & Sessions)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 19.1 | Аутентификация (/login, /logout) | 🟡 Partial | AuthController.swift, RESTServer.swift | Управление процессом входа через OAuth 2.0 провайдера (Anthropic, Google, Qwen). 2 |
| 19.2 | Обновление токенов (/oauth-refresh) | ❌ Planned | AuthController.swift, RESTServer.swift | Фоновый механизм ротации Refresh и Access токенов. 2 |
| 19.3 | Возобновление сессий (/resume) | ❌ Planned | AuthController.swift, RESTServer.swift | Восстановление предыдущей прерванной сессии из локального хранилища (memdir). 2 |
| 19.4 | Управление списком сессий (/session) | ❌ Planned | AuthController.swift, RESTServer.swift | Вывод каталога всех исторических разговоров (сортировка, переключение, удаление). 2 |
| 19.5 | Пакетное удаление | ❌ Planned | AuthController.swift, RESTServer.swift | Удаление множества старых сессий одной командой. 3 |
| 19.6 | Переименование и тегирование (/rename, /tag) | ❌ Planned | AuthController.swift, RESTServer.swift | Ручная привязка метаданных к сессиям для упрощения поиска. 2 |
| 19.7 | Экспорт и шеринг (/export, /share) | ❌ Planned | AuthController.swift, RESTServer.swift | Выгрузка логов переписки в файл или создание веб\-ссылки (Share Link). 2 |
| 19.8 | Многопользовательские аккаунты | ❌ Planned | AuthController.swift, RESTServer.swift | Поддержка нескольких профилей авторизации в одном терминале. |
| 19.9 | Очистка истории (/clear) | ✅ Done | AuthController.swift, RESTServer.swift | Удаление текущего контекста и очистка буферов диалога. 2 |
| 19.10 | Обработка падений аутентификации | ❌ Planned | AuthController.swift, RESTServer.swift | Классификатор FatalAuthenticationError для безопасной блокировки системы. 10 |
| 19.11 | Защита системного ключа | ❌ Planned | AuthController.swift, RESTServer.swift | Использование security.auth.selectedType и интеграция с Keytar. 7 |
| 19.12 | Управление жизненным циклом (Checkpointing) | ❌ Planned | AuthController.swift, RESTServer.swift | Сохранение дерева состояний на каждом шаге диалога. 7 |
