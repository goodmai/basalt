# Кластер 11: Безопасность, песочницы и Permissions

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 11.1 | Интерактивные подтверждения | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Запрос Y/N перед каждой деструктивной операцией (Default mode). 1 |
| 11.2 | Метод checkPermissions() | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Обязательная функция внутри каждого инструмента, возвращающая объект granted. 2 |
| 11.3 | Паттерны разрешений (Wildcards) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Правила маскирования доступа (например, Bash(git \*) или FileRead(\*)). 2 |
| 11.4 | Режим автоодобрения (BypassPermissions/Yolo) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Отключение запросов для доверенных сред или изолированных песочниц. 2 |
| 11.5 | Машинное обучение для разрешений (Auto) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Использование ML-классификатора для определения степени риска команды. 2 |
| 11.6 | Управление правилами (/permissions) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Команда для тонкой настройки политик доступа пользователем. 2 |
| 11.7 | Изоляция среды (Sandboxing) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Выполнение команд в Docker, Podman или Seatbelt (macOS). 2 |
| 11.8 | Инспекция контекста (Aegis-audit) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Глубокий поведенческий аудит навыков и инструментов перед вызовом. 41 |
| 11.9 | Политики доверенных папок (Trusted Folders) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Разделение прав доступа по директориям в settings.json. 7 |
| 11.10 | Защита аутентификации (OAuth gating) | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Хранение токенов авторизации в системных шифрованных хранилищах. 1 |
| 11.11 | Фильтрация логов телеметрии | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Удаление PII (Personal Identifiable Information) перед отправкой метрик. 6 |
| 11.12 | Превентивное сканирование ввода | ✅ Done | AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy | Анализ команд на Shell Injection перед передачей в bash. 22 |
