# Epic 14: Cloud Model Integration (OpenRouter API) ☁️

**Version:** v0.5.0  
**Priority:** HIGH  
**Effort:** 4 weeks (1 month)  
**Status:** Not started

## Overview

Enable hybrid inference architecture combining local MLX models with cloud models via OpenRouter API, providing seamless access to 100+ frontier models without downloading weights.

## Business Value

### Key Benefits
- **Hybrid Architecture:** Combine local privacy with cloud scale
- **Cost Optimization:** Local for dev, cloud for production
- **Model Access:** GPT-4, Claude 3.5, Gemini without local download
- **Automatic Failover:** Local → Cloud when OOM occurs
- **A/B Testing:** Compare local quantized vs cloud full-precision

### User Stories
1. As a developer, I want seamless switching between local and cloud models
2. As a user with 8GB RAM, I need access to 70B+ models via cloud
3. As a team lead, I want cost-effective GPT-4/Claude access
4. As a researcher, I need side-by-side local vs cloud comparison

## Strategic Rationale

```
┌─────────────────────────────────────────────────────────────┐
│                     GemmaServer v0.5.0                       │
│                    Hybrid Architecture                       │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │         ModelRouter (Smart)            │
        │  • Auto-fallback: Local → Cloud        │
        │  • Cost estimation & budget limits     │
        │  • RAM-based routing decisions         │
        └────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
    ┌──────────────┐                  ┌──────────────┐
    │ Local (MLX)  │                  │ Cloud (API)  │
    │              │                  │              │
    │ • Private    │                  │ • Scalable   │
    │ • Fast       │                  │ • Frontier   │
    │ • Free       │                  │ • Pay-per-use│
    │ • Limited    │                  │ • Unlimited  │
    └──────────────┘                  └──────────────┘
         MLX                          OpenRouter API
    Qwen 7B (local)              GPT-4, Claude, Gemini
```

## Task Breakdown

### 14.1 OpenRouter Client Implementation
**Effort:** 1 week | **Priority:** HIGH

**Deliverables:**
- `OpenRouterClient` actor with async/await
- API key management (env + keychain)
- Retry logic with exponential backoff
- Request/response DTOs (OpenAI-compatible)
- Rate limiting and quota tracking
- Error mapping to GemmaServerError
- **Dynamic Model Fetching (`GET /api/v1/models`)** to sync available models and exact pricing structure

**Files:**
- `Sources/GemmaServer/Cloud/OpenRouterClient.swift`
- `Tests/CloudTests/OpenRouterClientTests.swift`

**Key Features:**
```swift
actor OpenRouterClient {
    struct Config { ... }
    
    func chat(request: ChatRequest) async throws -> ChatResponse
    func getModels() async throws -> [OpenRouterModel]
    func getMetrics() -> (requests: Int, errors: Int)
}
```

**Testing:**
- ✅ API key validation
- ✅ Request/response parsing
- ✅ Retry logic on failures
- ✅ Timeout handling
- ✅ Error mapping (401, 429, 500)
- ⏳ Model fetching and pricing parsing

---

### 14.2 Model Router & Selection Strategy
**Effort:** 1 week | **Priority:** HIGH

**Deliverables:**
- `ModelRouter` actor with routing logic
- Model registry (local + cloud)
- Routing strategies: Auto, Local-only, Cloud-only, Hybrid
- Automatic fallback on OOM
- **Advanced OpenRouter Routing**: utilizing `models` array for auto-failover and `provider` preferences
- Cost estimation before cloud calls (using fetched pricing data)

**Files:**
- `Sources/GemmaServer/Cloud/ModelRouter.swift`
- `Tests/CloudTests/ModelRouterTests.swift`

**Routing Decision Tree:**
```
Request arrives with model="gpt-4"
    │
    ├─► Strategy = Local-only?
    │   └─► ❌ Error: Cloud models disabled
    │
    ├─► Strategy = Cloud-only?
    │   └─► ✅ Route to OpenRouter
    │
    ├─► Model in cloud-only list? (gpt-*, claude-*, etc)
    │   └─► ✅ Route to OpenRouter
    │
    ├─► Model available locally?
    │   │
    │   ├─► Sufficient RAM?
    │   │   └─► ✅ Route to Local MLX
    │   │
    │   └─► Insufficient RAM + Cloud available?
    │       └─► ✅ Fallback to Cloud
    │
    └─► Unknown model
        └─► Try Cloud (if configured)
```

**Key Methods:**
```swift
actor ModelRouter {
    enum RoutingStrategy {
        case auto, localOnly, cloudOnly, hybrid
    }
    
    func route(
        request: GenerationRequest,
        preferredModel: String?
    ) async throws -> GenerationResponse
    
    func estimateCost(
        model: String,
        tokens: Int
    ) -> Double
}
```

---

### 14.3 Cost Tracking & Budget Limits
**Effort:** 3 days | **Priority:** MEDIUM

**Deliverables:**
- `CostTracker` actor for usage monitoring
- Daily/monthly budget enforcement
- Cost warnings at 80% consumption
- Per-model cost breakdown
- Export to CSV/JSON

**Files:**
- `Sources/GemmaServer/Cloud/CostTracker.swift`
- `~/.gemmaserver/cloud/usage.json`

**Budget Protection:**
```swift
actor CostTracker {
    struct Budget {
        let dailyLimit: Double    // USD
        let monthlyLimit: Double  // USD
    }
    
    func recordRequest(...) async throws {
        let cost = calculateCost(...)
        
        // Hard stop at budget
        if dailyUsage + cost > budget.dailyLimit {
            throw .budgetExceeded
        }
        
        // Warning at 80%
        if dailyUsage > budget.dailyLimit * 0.8 {
            log("⚠️ 80% budget consumed")
        }
    }
}
```

**Pricing (OpenRouter typical):**
- GPT-4 Turbo: $0.01 input, $0.03 output (per 1K tokens)
- Claude 3.5: $0.003 input, $0.015 output
- Gemini Pro 1.5: $0.00125 input, $0.005 output

---

### 14.4 Streaming Support for Cloud Models
**Effort:** 1 week | **Priority:** MEDIUM

**Deliverables:**
- SSE streaming from OpenRouter
- Unified `AsyncStream<StreamChunk>` interface
- Error handling mid-stream
- Cancel support (abort generation)

**Integration:**
```swift
// Same interface for local and cloud
let stream = try await router.generateStream(
    request: request,
    model: "gpt-4"
)

for await chunk in stream {
    switch chunk {
    case .text(let token):
        print(token, terminator: "")
    case .metadata(let stats):
        print("\n[Cost: $\(stats.estimatedCost)]")
    }
}
```

---

### 14.5 CLI & Configuration
**Effort:** 2 days | **Priority:** HIGH

**Deliverables:**
- `cloud` subcommand group
- API key configuration wizard
- Connection testing
- Usage/cost reporting

**CLI Commands:**
```bash
# Configure
export OPENROUTER_API_KEY="sk-or-v1-..."
GemmaServer cloud configure --budget-daily 10 --budget-monthly 100

# Test connection
GemmaServer cloud test
# ✅ Connected to OpenRouter API
# ✅ API key valid
# ✅ 23 models available

# List models
GemmaServer cloud models --filter frontier
# gpt-4-turbo     (128K ctx, $10/1M tokens)
# claude-3.5      (200K ctx, $3/1M tokens)
# gemini-pro-1.5  (1M ctx, $1.25/1M tokens)

# Check costs
GemmaServer cloud cost --period today
# Today: $2.34 / $10.00 (23.4%)
# Requests: 47
# Tokens: 234K

# Generate with cloud model
GemmaServer chat --model gpt-4
```

**Configuration File:**
```json
// ~/.gemmaserver/cloud.json
{
  "apiKey": "***",
  "strategy": "auto",
  "budget": {
    "dailyLimit": 10.0,
    "monthlyLimit": 100.0
  },
  "preferredModels": {
    "gpt-4": "openai/gpt-4-turbo",
    "claude": "anthropic/claude-3.5-sonnet"
  }
}
```

---

### 14.6 Documentation & Examples
**Effort:** 2 days | **Priority:** MEDIUM

**Deliverables:**
- Setup guide
- Cost optimization strategies
- Model comparison matrix
- Security best practices

**Documents:**
1. `docs/cloud/OPENROUTER_SETUP.md`
   - Getting API key
   - Configuration
   - Testing connection

2. `docs/cloud/COST_OPTIMIZATION.md`
   - Routing strategies
   - Budget management
   - Model selection guide

3. `docs/cloud/MODEL_COMPARISON.md`
   - Local vs Cloud tradeoffs
   - Performance benchmarks
   - Cost analysis

4. `docs/cloud/SECURITY.md`
   - API key storage (Keychain)
   - Environment variables
   - .gitignore rules

**Example Use Cases:**
```markdown
### Use Case 1: Development Workflow
**Scenario:** Code during day, A/B test at night

```bash
# Development: Use free local models
export GEMMA_ROUTING_STRATEGY=local-only
GemmaServer serve --model Qwen2.5-7B-4bit

# Production testing: Compare with GPT-4
export GEMMA_ROUTING_STRATEGY=auto
GemmaServer chat --model gpt-4 --compare-with Qwen2.5-7B-4bit
```

### Use Case 2: Cost-Aware Routing
**Scenario:** Use local when possible, cloud for complex queries

```bash
# Auto-routing based on RAM
GemmaServer serve --routing auto --budget-daily 5.00

# Simple query → routes to local Qwen 7B (free)
curl -X POST /api/v1/generate -d '{"prompt": "Hello"}'

# Complex query → routes to GPT-4 ($0.10)
curl -X POST /api/v1/generate -d '{"prompt": "...", "model": "gpt-4"}'
```
```

---

## Dependencies

### Required Before Start
- ✅ Epic 7.1: Streaming REST API (for SSE)
- ✅ Epic 8: Security Audit (for API key handling)
- ⚠️ Swift Crypto for Keychain storage

### External Dependencies
- OpenRouter account (https://openrouter.ai)
- API key (free tier available)
- URLSession or AsyncHTTPClient

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| API key leakage | HIGH | Keychain storage, never commit, .gitignore |
| Unexpected costs | HIGH | Hard budget limits, 80% warnings |
| Privacy concerns | MEDIUM | Local-first default, explicit cloud opt-in |
| API reliability | MEDIUM | Retry logic, fallback to local |
| Rate limiting | LOW | Exponential backoff, quota tracking |

---

## Success Metrics

### Technical Metrics
- [ ] 99%+ cloud API success rate
- [ ] <500ms cloud latency (median)
- [ ] Zero API key leaks
- [ ] Budget enforcement 100% effective

### Business Metrics
- [ ] 50%+ users enable cloud integration
- [ ] Average cost <$5/user/month
- [ ] 3x model variety (local + cloud)
- [ ] 90% user satisfaction with hybrid mode

---

## Future Enhancements (v0.6.0+)

1. **Multi-Provider Support**
   - Direct OpenAI API
   - Direct Anthropic API
   - Azure OpenAI

2. **Advanced Routing**
   - Latency-based routing
   - Quality-based routing
   - Cost-per-quality optimization

3. **Caching Layer**
   - Semantic cache (similar prompts)
   - Response deduplication
   - Cost savings 30-50%

4. **Team Features**
   - Shared budget pools
   - Usage analytics per user
   - Admin dashboard

---

## Timeline

```
Week 1-2:  Task 14.1 (OpenRouter Client) + 14.5 (CLI)
Week 3:    Task 14.2 (Model Router) + 14.3 (Cost Tracking)
Week 4:    Task 14.4 (Streaming) + 14.6 (Documentation)
```

**Target Release:** v0.5.0 (August 2026)

---

## References

- OpenRouter API: https://openrouter.ai/docs
- OpenRouter Models: https://openrouter.ai/models
- OpenRouter Pricing: https://openrouter.ai/pricing
- API Key: https://openrouter.ai/keys
