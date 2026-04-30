# Epic 19: API & MCP Documentation 📖

Этот эпик посвящен полной документации всех интерфейсов приложения: REST API (A2A) и MCP (Model Context Protocol).

## 1. REST API (A2A)

### 1.1 Аутентификация
**Эндпоинт**: `POST /api/v1/auth/login`
**Описание**: Получение JWT токена для доступа к защищенным эндпоинтам.
**Запрос**:
```json
{
  "username": "admin",
  "password": "your_password"
}
```
**Ответ**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### 1.2 Генерация текста (Blocking)
**Эндпоинт**: `POST /api/v1/generate`
**Описание**: Генерация ответа модели в блокирующем режиме.
**Заголовок**: `Authorization: Bearer <token>`
**Запрос**:
```json
{
  "prompt": "Hello",
  "maxTokens": 100,
  "temperature": 0.7,
  "topP": 0.9
}
```

### 1.3 Генерация текста (Streaming SSE)
**Эндпоинт**: `POST /api/v1/generate/stream`
**Описание**: Потоковая генерация ответа через Server-Sent Events.
**Ответ**: Серия событий `data: {...}` завершающаяся `data: [DONE]`.

### 1.4 Проверка здоровья (Healthcheck)
**Эндпоинт**: `GET /api/v1/health`
**Описание**: Статус сервера и готовность модели.

---

## 2. MCP (Model Context Protocol)

### 2.1 Подключение (stdio)
Приложение работает как MCP сервер через стандартные потоки ввода-вывода (stdin/stdout). Каждое сообщение — это JSON-RPC 2.0 объект.

### 2.2 Доступные инструменты (Tools)

#### `gemma_generate`
- **Описание**: Генерация текста через локальную модель.
- **Аргументы**: `prompt`, `maxTokens`, `temperature`, `topP`.

#### `gemma_status`
- **Описание**: Возвращает статус готовности и версию сервера.

#### `playwright_screenshot`
- **Описание**: Захват скриншота любого веб-сайта.
- **Аргументы**: `url`, `width`, `height`.

---

## 3. CLI Команды и Режимы

### 3.1 Модели (Models)
- `Gemm models download <repo_id>`: Загрузка модели из HuggingFace.
- `Gemm models list`: Список локально доступных моделей.

### 3.2 Чат (Chat)
- `Gemm chat --model <id>`: CLI интерактивный чат.
- `Gemm chat --ui`: Запуск Rainbow Metal GUI.
- `Gemm chat --agent-real`: Запуск автоматизированной тестовой сюиты в GUI.

### 3.3 Скриншоты в GUI
В режиме GUI доступна команда `/screenshot <path>` прямо в поле ввода для захвата текущего состояния интерфейса.

---

## 4. Добавление скиллов (Skills)
На данный момент добавление новых MCP инструментов (скиллов) осуществляется через модификацию `Sources/Gem/MCP/MCPServer.swift`. В будущем планируется динамическая загрузка Swift-скриптов.
