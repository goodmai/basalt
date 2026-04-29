# Кластер 9: Многоагентная оркестрация (Agent Teams & Swarms)

| № | Функция (Feature) | Статус | Детали реализации | Что сделано |
| :--- | :--- | :---: | :--- | :--- |
| 9.1 | Спавн суб-агентов (AgentTool) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Создание дочернего агента с ограниченным скоупом и отдельным контекстом. 2 |
| 9.2 | Управление роями (TeamCreateTool) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Формирование групп агентов для параллельного исполнения подзадач. 2 |
| 9.3 | Коммуникация (SendMessageTool) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Обмен сообщениями и состоянием между независимыми агентами. 2 |
| 9.4 | Интерфейс мониторинга (/agents) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Управление жизненным циклом запущенных агентов пользователем. 2 |
| 9.5 | Протокол A2A (Agent2Agent) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Использование открытого стандарта Google для кросс-вендорного общения. 32 |
| 9.6 | Карточки агентов (Agent Cards) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | JSON-манифесты способностей, публикуемые для дискавери агентов. 32 |
| 9.7 | Освобождение ресурсов (TeamDeleteTool) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Уничтожение суб-агентов после выполнения задачи. 2 |
| 9.8 | Передача артефактов (A2A Artifacts) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Возврат структурированных результатов работы (файлы, блобы). 33 |
| 9.9 | Изоляция памяти (Context Opacity) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Сокрытие внутренних промптов и памяти суб-агентов от оркестратора. 34 |
| 9.10 | Иерархическая маршрутизация | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Паттерн Triage/Router для классификации интентов на входе. 37 |
| 9.11 | Аукционная маршрутизация | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Суб-агенты вычисляют скоринг своей уверенности в решении задачи. 38 |
| 9.12 | Согласование модальностей (Negotiation) | ✅ Done | ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift | Динамическое согласование форматов ответа (текст, iframe, форма). 34 |
