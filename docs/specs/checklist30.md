# Кластер 30: Облачные среды и инфраструктурная интеграция (Cloud & Infrastructure)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 30.1 | Интеграция с Cloud Shell | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Поставка Gemini CLI "из коробки" в Google Cloud Shell без дополнительных настроек. 6 |
| 30.2 | Инструмент Cloud Run MCP Server | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Использование MCP серверов, развернутых в бессерверных средах (Serverless). 67 |
| 30.3 | Enterprise Guide Deployments | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Развертывание CLI утилит с централизованным управлением политиками (MDM). 7 |
| 30.4 | Интеграция с BaaS (Firebase / Supabase) | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Использование специализированных плагинов (AlloyDB extension) для работы с Managed БД. 12 |
| 30.5 | Работа с инфраструктурным кодом (Terraform) | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Парсинг и валидация HCL-синтаксиса; генерация планов развертывания. |
| 30.6 | Взаимодействие с Kubernetes | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Инструменты вызова kubectl с парсингом YAML манифестов. |
| 30.7 | Поддержка x402 протокола (/x402) | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Встроенная поддержка микроплатежей для оплаты ресурсов (Claude Code). 2 |
| 30.8 | Оптимизация затрат на облако | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Анализ агентом метрик CloudWatch и предложение изменения инстансов (Right-sizing). |
| 30.9 | Интеграция с Vault | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Извлечение секретов из HashiCorp Vault во время выполнения скриптов (вместо .env). |
| 30.10 | Работа через бастион-хосты (SSH Tunneling) | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Установка SSH туннелей (/remote-env) для работы агента с закрытыми серверами. 2 |
| 30.11 | A2P транзакционные мандаты (AP2 Protocol) | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Использование мандатов (cryptographic signatures) для авторизации транзакций агентом. 60 |
| 30.12 | Сквозной аудит Enterprise уровня | ✅ Done | CloudAPIClient.swift, OpenRouterClient.swift | Интеграция с системами SIEM (Splunk) для централизованного логирования действий агентов. 69 |
