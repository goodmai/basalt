# GemmaServer Product Roadmap & Architecture Plan

**Product Vision:** Production-grade local LLM inference server for Apple Silicon, optimized for developer workflows and agent-to-agent communication.

**Target Users:** 
- Developers running local AI assistants (Cursor, Claude Desktop)
- Teams building agent-to-agent (A2A) systems
- Researchers benchmarking MLX models on Apple Silicon

**Current Status:** v0.1.0 — Core inference server operational, 4 verified models, 51/51 tests passing

---

## Version History & Semantic Versioning

**Versioning Strategy:** [Semantic Versioning 2.0.0](https://semver.org/)
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes, incompatible API changes
- **MINOR**: New features, backward-compatible functionality
- **PATCH**: Bug fixes, backward-compatible fixes

**Version Increment Rules:**
- **Epic completion** → MINOR version bump (new feature set)
- **Breaking change** → MAJOR version bump (API incompatibility)
- **Bug fix commit** → PATCH version bump (fix: prefix)
- **Feature commit** → MINOR version bump (feat: prefix)
- **Docs/chore commit** → No version bump

### Version Changelog

**v0.1.0** (2026-04-15) — Initial Release
- Epic 1: Production-Ready Inference Server ✅
- Features: 5 major features
  - Dual interface (MCP + REST)
  - JWT authentication
  - Performance benchmarking
  - Multi-model support (4 models)
  - Performance optimizations
- Commits: 15 commits
- Test Coverage: 51/51 tests passing

**v0.2.0** (Target: 2026-05-15 | **IN PROGRESS - 72% Complete**) — Security & Streaming
- Epic 2: Context Window Profiling ✅ (2 features)
  - Context degradation profiler ✅
  - Dynamic token budgeting ✅
- Epic 8: Security Audit ✅ (2 features) **COMPLETED**
  - Dependency security audit ✅ (10/10 score)
  - CWE security audit ✅ (0 critical vulnerabilities)
- Epic 3.1: Interactive Onboarding ✅ **COMPLETED**
  - System profiler with M1-M5 detection ✅
  - Interactive CLI setup ✅
- Epic 7.1: Streaming REST API ✅ **COMPLETED**
  - Server-Sent Events endpoint ✅
  - Web demo client ✅
- Target Commits: +20 commits (Achieved: 34 commits)
- Target Test Coverage: 100% unit coverage (Achieved: 101 tests passing)
- Security Score: 10/10 🏆

**v0.3.0** (Target: 2026-06-15) — Documentation & Model Ecosystem
- Epic 4: Documentation Website (3 features)
- Epic 3: Onboarding & Benchmarking (3 features)
  - Interactive onboarding flow
  - SWE benchmark suite
  - Scheduled benchmark execution
  - Documentation framework setup
  - Automated documentation deployment
  - Documentation quality checks
- Epic 5: Model Compatibility (3 features)
  - Model compatibility matrix
  - Automated model testing
  - Model validation
- Epic 6: Developer Experience (3 features)
  - Homebrew installation
  - Interactive model selection
  - Rich terminal output
- Target Commits: +25 commits

**v0.4.0** (Target: 2026-07-15) — Automation & Remote Control
- Epic 11: Scheduled Tasks & Multi-Agent Orchestration (3 features)
  - Task scheduler & cron system
  - Multi-agent orchestration (Gemini, Claude, Copilot, etc.)
  - Agent collaboration & consensus
- Epic 12: Telegram Bot Integration (3 features)
  - Telegram bot setup & authentication
  - Remote session management
  - Inline queries & notifications
- Epic 13: Session Analytics (1 feature)
  - Session analytics & beautiful exit summary (/quit)
- Target Commits: +40 commits


**v0.5.0** (Target: 2026-08-15) — Cloud Integration & Hybrid Inference
- Epic 14: Cloud Model Integration (OpenRouter API) (6 features)
  - [x] OpenRouter API client with authentication
  - [ ] Fetch available models dynamically (`GET /api/v1/models`)
  - [x] Model router with auto-fallback (local → cloud)
  - [ ] Advanced OpenRouter fallbacks (using `models` array) and provider preferences
  - Cost tracking and budget limits
  - Streaming support for cloud models
  - CLI configuration and testing
  - Documentation and examples
- Epic 9: MCP Plugin Marketplace (3 features)
  - MCP server discovery
  - Agent capability analysis
  - Marketplace UI
- Epic 13: Usage Billing (2 features)
  - Usage tracking & billing integration
  - Analytics dashboard
- Target Commits: +40 commits
- Key Features: Hybrid local+cloud inference, cost optimization

**v0.6.0** (Target: 2026-09-15) — Advanced Cloud Features

**v1.0.0** (Target: 2026-09-01) — Production Release
- Epic 7: App Store Distribution (4 features)
  - App Store preparation
  - Sandboxing & entitlements
  - Code signing & notarization
  - App Store submission
- All epics completed
- Zero HIGH/CRITICAL vulnerabilities
- 100% unit test coverage
- Target Total Commits: 100+

### Feature Counter

**Completed Features:** 7
- ✅ Dual interface architecture
- ✅ JWT authentication
- ✅ Performance benchmarking
- ✅ Multi-model support
- ✅ Performance optimizations
- ✅ Context degradation profiler
- ✅ Dynamic token budgeting

**In Progress Features:** 6
- 🔄 Model compatibility matrix
- 🔄 Automated model testing
- 🔄 Dependency security audit
- 🔄 CWE security audit
- 🔄 Privacy compliance audit
- 🔄 Homebrew installation

**Planned Features:** 28
- 📋 Documentation framework setup
- 📋 Interactive onboarding flow
- 📋 SWE benchmark suite (5 tasks)
- 📋 Scheduled benchmark execution
- 📋 Task scheduler & cron system
- 📋 Multi-agent orchestration
- 📋 Agent collaboration & consensus
- 📋 Telegram bot setup
- 📋 Remote session management
- 📋 Inline queries & notifications
- 📋 Session analytics & exit summary (/quit) 🆕
- 📋 Usage tracking & billing integration 🆕
- 📋 Analytics dashboard 🆕
- 📋 Automated documentation deployment
- 📋 Documentation quality checks
- 📋 Model validation
- 📋 Interactive model selection
- 📋 Rich terminal output
- 📋 MCP server discovery
- 📋 Agent capability analysis
- 📋 Marketplace UI
- 📋 App Store preparation
- 📋 Sandboxing & entitlements
- 📋 Code signing & notarization
- 📋 App Store submission
- 📋 Streaming REST API
- 📋 Multi-model serving
- 📋 Quantization experiments

**Total Features:** 41 (7 completed, 6 in progress, 28 planned)

### Commit Counter

**Total Commits:** 20 (as of 2026-04-28)

**By Type:**
- `feat:` 12 commits (60%) → MINOR version bumps
- `fix:` 2 commits (10%) → PATCH version bumps
- `docs:` 4 commits (20%) → No version bump
- `chore:` 1 commit (5%) → No version bump
- `security:` 1 commit (5%) → MINOR version bump

**By Epic:**
- Epic 1: 15 commits
- Epic 2: 3 commits
- Epic 5: 2 commits

**Next Version Calculation:**
- Current: v0.1.0
- Completed features since v0.1.0: 2 (context profiler, token budgeting)
- Next version: v0.2.0 (when Epic 2 + Epic 5 security audit complete)

---

## Epic 1: Production-Ready Inference Server ✅ (COMPLETED)

**Business Value:** Stable, performant dual-interface server (MCP + REST) with authentication, supporting multiple model families.

**User Stories:**
- As a developer, I want to run local LLM inference without cloud dependencies
- As a team, we need JWT-authenticated REST API for microservices integration
- As a researcher, I need accurate performance metrics (TPS, TTFT, memory)

### Completed Tasks
- [x] **1.1 Dual Interface Architecture**
  - MCP stdio server (JSON-RPC 2.0) for IDE integration
  - REST HTTP server (Hummingbird 2.x) on port 8080
  - Shared `ModelOrchestratorActor` — single source of truth, FIFO queue
  - **Value:** One inference engine serves both transports → no resource duplication

- [x] **1.2 Authentication & Security**
  - JWT-based auth with SQLite session store
  - Password hashing (SHA256 + salt)
  - Token revocation (blocklist)
  - **Value:** Production-ready security for team deployments

- [x] **1.3 Performance Benchmarking**
  - `PerformanceBenchmark` CLI tool
  - Metrics: TPS, TTFT, generation time, memory usage
  - Warmup iteration + statistical analysis (avg, min, max, σ)
  - **Value:** Data-driven model selection for hardware constraints

- [x] **1.4 Multi-Model Support**
  - HuggingFace Hub integration with resume-capable downloads
  - Automatic cache resolution (`~/.cache/huggingface/hub/`)
  - Verified models: Gemma 4, Qwen3.5, Qwen3.6, Qwen2.5-Coder
  - **Value:** Flexibility to choose model by use case (speed vs quality vs code)

- [x] **1.5 Performance Optimizations**
  - O(n²) → O(n) string concatenation in streaming
  - Removed double JSON serialization in MCP
  - Duration helper extension for timing
  - ModelCache refs/main resolution for standard HF layout
  - **Value:** 2-3x faster response assembly for long generations

---

## Epic 2: Context Window Profiling & Optimization 🔄 (IN PROGRESS)

**Business Value:** Understand and optimize performance degradation as context grows. Enable users to make informed decisions about context budget vs speed.

**User Stories:**
- As a developer, I want to know the TPS/TTFT curve for 1k-128k token contexts
- As an agent builder, I need to budget context dynamically based on available RAM
- As a researcher, I want reproducible benchmarks for context degradation

### 2.1 Context Degradation Profiler
**Status:** COMPLETED  
**Priority:** HIGH  
**Effort:** 2-3 days

**Acceptance Criteria:**
- [x] `ContextDegradationProfiler` actor with configurable context sizes
- [x] Automated benchmark: 1k, 4k, 16k, 32k, 64k, 128k tokens
- [x] Export results to JSON: `{contextSize, tps, ttft, memoryMB}`
- [x] CLI command: `GemmaServer profile-context --model <id> --output results.json`

**Technical Design:**
```swift
actor ContextDegradationProfiler {
    let orchestrator: ModelOrchestratorActor
    
    func profile(contextSizes: [Int]) async throws -> [ContextBenchmark] {
        var results: [ContextBenchmark] = []
        for size in contextSizes {
            let prompt = generatePrompt(tokenCount: size)
            let response = try await orchestrator.generate(request: .init(prompt: prompt))
            results.append(ContextBenchmark(
                contextSize: size,
                tps: response.tokensPerSecond,
                ttft: response.timeToFirstToken,
                memoryMB: response.memory.activeBytes / 1024 / 1024
            ))
        }
        return results
    }
}
```

**Test Plan:**
1. Unit test: Mock orchestrator, verify correct prompt generation
2. Integration test: Real model (Qwen3.5-4B), verify JSON export
3. Benchmark: Run on 3 models (4B, 9B, 27B), compare curves

**Workflow:**
1. TDD: Write test for `profile(contextSizes:)` with mock
2. Implement: `ContextDegradationProfiler` actor
3. Integration: CLI command `profile-context`
4. Profile: Run on Qwen3.5-4B, export JSON
5. Commit: `feat: Add context degradation profiler`

---

### 2.2 Dynamic Token Budgeting
**Status:** COMPLETED  
**Priority:** MEDIUM  
**Effort:** 1-2 days

**Business Value:** Automatically adjust `maxTokens` based on available RAM to prevent OOM crashes.

**User Story:**
- As a user with 16GB RAM, I want the server to auto-limit context to prevent crashes
- As a developer, I want a safety margin (e.g., reserve 4GB for OS)

**Acceptance Criteria:**
- [x] Function: `calculateMaxTokens(availableRAM: Int, modelSize: Int) -> Int`
- [x] Use `host_statistics64()` (macOS alternative to `os_proc_available_memory()`) to query free RAM
- [x] Apply safety margin (default 20%)
- [x] Log warning if user requests exceed budget

**Technical Design:**
```swift
func calculateMaxTokens(availableRAM: Int64, modelSizeMB: Int) -> Int {
    let safetyMargin = 0.8  // Reserve 20% for OS
    let usableRAM = Int64(Double(availableRAM) * safetyMargin)
    let availableForContext = usableRAM - Int64(modelSizeMB * 1024 * 1024)
    
    // Rough estimate: 1 token ≈ 2 bytes in KV cache (FP16)
    let maxTokens = Int(availableForContext / 2)
    return min(maxTokens, 128_000)  // Cap at 128k
}
```

**Test Plan:**
1. Unit test: Various RAM scenarios (8GB, 16GB, 32GB, 64GB)
2. Integration test: Load 27B model on 24GB machine, verify limit
3. Stress test: Attempt to exceed limit, verify graceful rejection

**Workflow:**
1. TDD: Write test for `calculateMaxTokens`
2. Implement: Add to `ServerConfig` initialization
3. Integration: Apply limit in `ModelOrchestratorActor.generate`
4. Test: Verify OOM prevention on 27B model
5. Commit: `feat: Add dynamic token budgeting based on available RAM`

---

## Epic 6: Model Compatibility & Testing 🧪 (IN PROGRESS)
---

## Epic 4: Documentation Website & Deployment 📚
---

## Epic 3: Onboarding & Automated Benchmarking 🚀

**Business Value:** First-run experience that automatically profiles hardware and recommends optimal models, reducing setup friction and ensuring best performance.

**User Stories:**
- As a new user, I want the app to guide me through initial setup
- As a developer, I need to know which models will run well on my hardware
- As a user, I want to benchmark models overnight without blocking my work
- As a team lead, I need performance data to choose the right model for deployment

### 3.1 Interactive Onboarding Flow
**Status:** ✅ COMPLETED  
**Priority:** HIGH (for v0.3.0)  
**Effort:** 1 week (Completed in 1 day)

**Business Value:** Guided onboarding reduces time-to-first-inference from 30 minutes to 5 minutes.

**Acceptance Criteria:**
- [x] Detect first launch (no config file exists) ✅
- [x] System resource detection (RAM, GPU, CPU cores, disk space) ✅
- [x] Interactive dialog with 3 options: Start Now, Skip, Custom ✅
- [x] Model recommendation based on available RAM ✅
- [ ] Download recommended model with progress bar ⚠️ (deferred to 3.2)
- [ ] Run first inference test ⚠️ (deferred to 3.2)
- [x] Save onboarding completion state ✅

**Completed Features:**
* ✅ SystemProfiler actor with M1/M2/M3/M4/M5 detection
* ✅ OnboardingFlow actor with beautiful ASCII art UI
* ✅ OnboardCommand CLI integration
* ✅ 5-tier model recommendations by RAM:
  - <8GB: Qwen 2.5 0.5B (~512 MB)
  - 8-16GB: Qwen 2.5 3B (~2.3 GB)
  - 16-32GB: Qwen 2.5 7B (~5.8 GB)
  - 32-64GB: Qwen 2.5 14B (~9.5 GB)
  - 64GB+: Qwen 2.5 32B (~19 GB)
* ✅ State persistence to ~/.gemmaserver/onboarding.json

**CLI Commands:**
```bash
GemmaServer onboard                # Full interactive setup
GemmaServer onboard --profile-only # System info only
GemmaServer onboard --reset        # Re-run setup
        let availableRAM: Int64      // bytes
        let cpuCores: Int
        let gpuName: String
        let gpuMemory: Int64         // bytes
        let diskSpace: Int64         // bytes available
        let osVersion: String
        let chipModel: String        // M1, M2, M3, M4, M5
    }
    
    func detectResources() async -> SystemResources {
        // Use sysctl, Metal API, FileManager
        let totalRAM = ProcessInfo.processInfo.physicalMemory
        let cpuCores = ProcessInfo.processInfo.processorCount
        
        // Metal GPU detection
        let device = MTLCreateSystemDefaultDevice()
        let gpuName = device?.name ?? "Unknown"
        let gpuMemory = device?.recommendedMaxWorkingSetSize ?? 0
        
        // Disk space
        let fileURL = URL(fileURLWithPath: NSHomeDirectory())
        let values = try? fileURL.resourceValues(forKeys: [.volumeAvailableCapacityKey])
        let diskSpace = values?.volumeAvailableCapacity ?? 0
        
        return SystemResources(
            totalRAM: Int64(totalRAM),
            availableRAM: getAvailableRAM(),
            cpuCores: cpuCores,
            gpuName: gpuName,
            gpuMemory: Int64(gpuMemory),
            diskSpace: Int64(diskSpace),
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            chipModel: detectChipModel()
        )
    }
    
    func recommendModel(resources: SystemResources) -> String {
        let ramGB = resources.totalRAM / 1_073_741_824
        
        return switch ramGB {
        case 0..<8:   "mlx-community/Qwen3.5-4B-4bit"      // 2.3 GB RAM
        case 8..<16:  "mlx-community/Qwen3.5-9B-OptiQ-4bit" // 5.8 GB RAM
        case 16..<32: "mlx-community/Qwen3.6-27B-4bit"     // 14.5 GB RAM
        default:      "mlx-community/Qwen3.6-27B-4bit"     // Best quality
        }
    }
}
```

**Interactive Dialog:**
```swift
actor OnboardingFlow {
    func run() async throws {
        print("🚀 Welcome to GemmaServer!")
        print("\nDetecting system resources...")
        
        let profiler = SystemProfiler()
        let resources = await profiler.detectResources()
        
        print("\n📊 System Profile:")
        print("  RAM:  \(resources.totalRAM / 1_073_741_824) GB")
        print("  GPU:  \(resources.gpuName)")
        print("  CPU:  \(resources.cpuCores) cores")
        print("  Chip: \(resources.chipModel)")
        
        let recommended = await profiler.recommendModel(resources: resources)
        print("\n✨ Recommended model: \(recommended)")
        print("   Expected performance: ~\(estimateTPS(model: recommended)) TPS")
        
        print("\n🎯 Benchmark Options:")
        print("  1. Start benchmark now (~15 minutes)")
        print("  2. Skip benchmark (use defaults)")
        print("  3. Schedule for tonight (22:00)")
        print("\nChoice (1/2/3): ", terminator: "")
        
        guard let choice = readLine() else { return }
        
        switch choice {
        case "1":
            try await runBenchmarkNow(model: recommended)
        case "2":
            print("⏭️  Skipping benchmark. Using default settings.")
            try await downloadAndTest(model: recommended)
        case "3":
            try await scheduleBenchmark(model: recommended, time: "22:00")
        default:
            print("Invalid choice. Defaulting to skip.")
            try await downloadAndTest(model: recommended)
        }
    }
}
```

**Workflow:**
1. TDD: Write tests for SystemProfiler
2. Implement: Resource detection
3. Implement: Interactive dialog
4. Test: Run on different hardware (M1, M2, M3, M4, M5)
5. Commit: `feat: Add interactive onboarding with system profiling`

---

### 3.2 SWE Benchmark Suite (5 Real Tasks)
**Status:** Not started  
**Priority:** HIGH  
**Effort:** 1 week

**Business Value:** Real-world task benchmarks provide actionable performance data, not just synthetic metrics.

**Acceptance Criteria:**
- [ ] 5 real software engineering tasks
- [ ] Time-based evaluation (not just TPS)
- [ ] Quality scoring (correctness, completeness)
- [ ] Performance report with recommendations
- [ ] Export results to JSON

**5 SWE Benchmark Tasks:**

**Task 1: Code Generation (Simple)**
```
Prompt: "Write a Swift function to calculate Fibonacci numbers recursively."
Expected: Working Swift code with proper syntax
Time Budget: 10 seconds
Quality Metrics: Compiles, correct algorithm, handles edge cases
```

**Task 2: Code Refactoring**
```
Prompt: "Refactor this code to use async/await instead of callbacks: [code snippet]"
Expected: Correct async/await transformation
Time Budget: 15 seconds
Quality Metrics: Maintains functionality, proper error handling, Swift 6 compliant
```

**Task 3: Bug Fix**
```
Prompt: "Fix the race condition in this actor: [code snippet]"
Expected: Identifies issue, provides correct fix
Time Budget: 20 seconds
Quality Metrics: Correct diagnosis, working fix, explanation
```

**Task 4: API Design**
```
Prompt: "Design a REST API for a todo list app with CRUD operations."
Expected: Complete API spec with endpoints, methods, payloads
Time Budget: 30 seconds
Quality Metrics: RESTful design, proper HTTP methods, error handling
```

**Task 5: Architecture Review**
```
Prompt: "Review this system architecture and suggest improvements: [diagram]"
Expected: Identifies bottlenecks, suggests optimizations
Time Budget: 45 seconds
Quality Metrics: Actionable suggestions, considers scalability, security
```

**Benchmark Implementation:**
```swift
actor SWEBenchmark {
    struct Task: Codable, Sendable {
        let id: String
        let name: String
        let prompt: String
        let timeBudget: TimeInterval
        let expectedKeywords: [String]
    }
    
    struct Result: Codable, Sendable {
        let taskId: String
        let timeElapsed: TimeInterval
        let tokensGenerated: Int
        let qualityScore: Double  // 0.0-1.0
        let passed: Bool
    }
    
    let tasks: [Task] = [
        Task(id: "code_gen", name: "Code Generation", 
             prompt: "Write a Swift function to calculate Fibonacci...",
             timeBudget: 10.0,
             expectedKeywords: ["func", "fibonacci", "return"]),
        // ... 4 more tasks
    ]
    
    func runBenchmark(orchestrator: ModelOrchestratorActor) async throws -> [Result] {
        var results: [Result] = []
        
        for task in tasks {
            print("Running: \(task.name)...")
            let start = ContinuousClock.now
            
            let response = try await orchestrator.generate(
                request: .init(prompt: task.prompt, maxTokens: 500)
            )
            
            let elapsed = start.duration(to: .now).inSeconds
            let quality = evaluateQuality(
                response: response.generatedText,
                expected: task.expectedKeywords
            )
            
            results.append(Result(
                taskId: task.id,
                timeElapsed: elapsed,
                tokensGenerated: response.completionTokens,
                qualityScore: quality,
                passed: elapsed <= task.timeBudget && quality >= 0.7
            ))
        }
        
        return results
    }
    
    func generateReport(results: [Result]) -> String {
        let passed = results.filter(\.passed).count
        let avgTime = results.map(\.timeElapsed).reduce(0, +) / Double(results.count)
        let avgQuality = results.map(\.qualityScore).reduce(0, +) / Double(results.count)
        
        return """
        📊 SWE Benchmark Report
        
        Tasks Passed: \(passed)/\(results.count)
        Avg Time:     \(String(format: "%.1f", avgTime))s
        Avg Quality:  \(String(format: "%.0f", avgQuality * 100))%
        
        Recommendation: \(getRecommendation(avgTime: avgTime, avgQuality: avgQuality))
        """
    }
}
```

**Workflow:**
1. Design: Define 5 real SWE tasks
2. Implement: SWEBenchmark actor
3. Implement: Quality evaluation logic
4. Test: Run on all verified models
5. Commit: `feat: Add SWE benchmark suite with 5 real tasks`

---

### 3.3 Scheduled Benchmark Execution
**Status:** Not started  
**Priority:** MEDIUM  
**Effort:** 3 days

**Business Value:** Overnight benchmarking doesn't interrupt user's work.

**Acceptance Criteria:**
- [ ] Schedule benchmark for specific time (e.g., 22:00)
- [ ] Run benchmark in background
- [ ] Send notification when complete
- [ ] Save results to file
- [ ] Resume if interrupted

**Technical Design:**
```swift
actor BenchmarkScheduler {
    func schedule(time: String, model: String) async throws {
        let components = time.split(separator: ":")
        guard components.count == 2,
              let hour = Int(components[0]),
              let minute = Int(components[1]) else {
            throw SchedulerError.invalidTime
        }
        
        var dateComponents = Calendar.current.dateComponents([.year, .month, .day], from: Date())
        dateComponents.hour = hour
        dateComponents.minute = minute
        
        guard let scheduledDate = Calendar.current.date(from: dateComponents) else {
            throw SchedulerError.invalidDate
        }
        
        // If time already passed today, schedule for tomorrow
        let targetDate = scheduledDate < Date() 
            ? Calendar.current.date(byAdding: .day, value: 1, to: scheduledDate)!
            : scheduledDate
        
        print("✅ Benchmark scheduled for \(targetDate)")
        
        // Use Timer or DispatchQueue.asyncAfter
        let delay = targetDate.timeIntervalSinceNow
        Task {
            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            try await runScheduledBenchmark(model: model)
        }
    }
    
    func runScheduledBenchmark(model: String) async throws {
        print("🌙 Starting scheduled benchmark...")
        
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine())
        try await orchestrator.loadModel(path: ModelCache.cacheDir(for: model).path)
        
        let benchmark = SWEBenchmark()
        let results = try await benchmark.runBenchmark(orchestrator: orchestrator)
        
        // Save results
        let report = benchmark.generateReport(results: results)
        let url = URL(fileURLWithPath: "benchmark_results_\(Date().ISO8601Format()).json")
        try JSONEncoder().encode(results).write(to: url)
        
        // Send notification
        sendNotification(title: "Benchmark Complete", body: report)
        
        print("✅ Benchmark complete. Results saved to \(url.path)")
    }
}
```

**Workflow:**
1. Implement: BenchmarkScheduler actor
2. Implement: macOS notification support
3. Test: Schedule for 1 minute in future
4. Test: Resume after app restart
5. Commit: `feat: Add scheduled benchmark execution`

---

**Business Value:** Professional documentation website increases adoption, reduces support burden, and establishes credibility.

**User Stories:**
- As a new user, I want comprehensive documentation to get started quickly
- As a developer, I need API reference docs with code examples
- As a contributor, I want to understand the architecture before contributing
- As a team lead, I need to share documentation links with my team

### 4.1 Documentation Framework Setup
**Status:** Not started  
**Priority:** HIGH (for v0.3.0)  
**Effort:** 1 week

**Business Value:** Automated documentation generation from code comments reduces maintenance burden.

**Acceptance Criteria:**
- [ ] Choose documentation framework (Swift-DocC, Publish, or Jazzy)
- [ ] Set up documentation structure
- [ ] Configure automated doc generation from source code
- [ ] Add code examples and tutorials
- [ ] Set up local preview server

**Framework Options:**

**Option 1: Swift-DocC (Recommended)**
- **Pros:** Official Apple tool, Xcode integration, modern design, symbol linking
- **Cons:** Requires Xcode, macOS-only build
- **Use case:** API reference, tutorials, articles
```bash
# Generate DocC archive
swift package generate-documentation --target GemmaServer

# Preview locally
swift package --disable-sandbox preview-documentation --target GemmaServer
```

**Option 2: Publish by John Sundell**
- **Pros:** Pure Swift, flexible, Markdown-based, custom themes
- **Cons:** Manual setup, less automated than DocC
- **Use case:** Blog-style docs, custom layouts

**Option 3: Jazzy**
- **Pros:** Mature, Apple-style docs, SourceKitten integration
- **Cons:** Ruby dependency, less modern than DocC
- **Use case:** API reference only

**Recommended: Swift-DocC + GitHub Pages**

**Documentation Structure:**
1. **Getting Started** - Installation, Quick start, First inference
2. **Guides** - CLI commands, REST API, MCP integration, Model selection
3. **Tutorials** - Building chatbot, Claude Desktop integration, Performance optimization
4. **API Reference** - Auto-generated from code comments
5. **Architecture** - System design, Actor concurrency, Performance, Security

**Workflow:**
1. Choose: Swift-DocC as framework
2. Setup: Create `.docc` bundle
3. Write: Documentation articles and tutorials
4. Generate: Build documentation archive
5. Commit: `feat: Add Swift-DocC documentation structure`

---

### 4.2 Automated Documentation Deployment
**Status:** Not started  
**Priority:** HIGH  
**Effort:** 3-5 days

**Business Value:** Automated deployment ensures docs are always up-to-date with latest code.

**Acceptance Criteria:**
- [ ] GitHub Actions workflow for doc generation
- [ ] Deploy to GitHub Pages on main branch push
- [ ] Custom domain setup (docs.gemmaserver.dev)
- [ ] HTTPS enabled
- [ ] Search functionality
- [ ] Version selector (v0.1.0, v0.2.0, latest)

**GitHub Actions Workflow:**
```yaml
name: Documentation
on:
  push:
    branches: [main]
    paths: ['Sources/**', 'docs/**']

jobs:
  build-and-deploy:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - name: Generate Documentation
        run: swift package generate-documentation --target GemmaServer
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
```

**Workflow:**
1. Setup: GitHub Pages in repository settings
2. Configure: Custom domain DNS
3. Implement: GitHub Actions workflow
4. Test: Deploy to staging branch first
5. Deploy: Enable for main branch
6. Commit: `feat: Add automated documentation deployment`

---

### 4.3 Documentation Quality & Maintenance
**Status:** Not started  
**Priority:** MEDIUM  
**Effort:** Ongoing

**Business Value:** High-quality docs reduce support requests and improve user satisfaction.

**Acceptance Criteria:**
- [ ] 100% public API documented
- [ ] All code examples tested in CI
- [ ] Documentation coverage report
- [ ] Broken link checker
- [ ] Spelling/grammar checker
- [ ] Documentation review process

**Documentation Coverage Tool:**
```bash
#!/bin/bash
# scripts/check-doc-coverage.sh
PUBLIC_SYMBOLS=$(swift-symbolgraph-extract -module-name GemmaServer)
DOCUMENTED=$(grep -r "///" Sources/ | wc -l)
COVERAGE=$((DOCUMENTED * 100 / PUBLIC_SYMBOLS))
echo "Documentation Coverage: ${COVERAGE}%"
[ $COVERAGE -lt 100 ] && exit 1
```

**Workflow:**
1. Implement: Documentation coverage checker
2. Add: Code example tests
3. Setup: Broken link checker
4. Create: Documentation review template
5. Commit: `feat: Add documentation quality checks`

---

**Business Value:** Ensure broad model support, document working configurations, provide clear error messages for incompatible models.

**User Stories:**
- As a user, I want to know which models work before downloading 15GB
- As a developer, I need clear error messages when a model fails to load
- As a contributor, I want automated tests for new model families

### 3.1 Model Compatibility Matrix
**Status:** Partially complete (4/10 tested models work)  
**Priority:** HIGH  
**Effort:** 1 week

**Acceptance Criteria:**
- [x] Test Qwen3.5 (4B, 9B OptiQ) ✅
- [x] Test Qwen2.5-Coder-7B ✅
- [x] Test Qwen3.6-27B ✅
- [x] Remove non-working models from cache ✅
- [ ] Investigate failures: Why do 35B MoE, Claude-distilled, DeepSeek fail?
- [ ] Document failure modes in README
- [ ] Add model validation before download

**Investigation Tasks:**
- [ ] Check `config.json` structure for failed models
- [ ] Verify tokenizer compatibility (AutoTokenizer vs custom)
- [ ] Test with different MLX versions (0.31.3 vs latest)
- [ ] Check for MoE-specific requirements (routing layers)

**Workflow:**
1. Debug: Load failed model manually, capture error logs
2. Fix: Implement workaround or document limitation
3. Test: Re-run automated test suite
4. Document: Update README with compatibility notes
5. Commit: `fix: Add MoE model support` or `docs: Document MoE limitations`

---

### 3.2 Automated Model Testing Pipeline
**Status:** COMPLETED  
**Priority:** MEDIUM  
**Effort:** 2-3 days

**Business Value:** Catch regressions early, verify new MLX versions don't break existing models.

**Acceptance Criteria:**
- [x] GitHub Actions workflow: test 4 verified models on each PR
- [x] Nightly job: test all models in cache
- [x] Performance regression detection: fail if TPS drops >10%
- [x] Artifact: JSON report with TPS/TTFT/memory for each model

**Technical Design:**
```yaml
# .github/workflows/model-tests.yml
name: Model Compatibility Tests
on: [pull_request, schedule]
jobs:
  test-models:
    runs-on: macos-14  # M1 runner
    steps:
      - uses: actions/checkout@v4
      - name: Download test models
        run: |
          swift run GemmaServer models download mlx-community/Qwen3.5-4B-4bit
      - name: Run tests
        run: swift test
      - name: Benchmark
        run: |
          swift run PerformanceBenchmark --model mlx-community/Qwen3.5-4B-4bit --iterations 5
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark_*.json
```

**Workflow:**
1. TDD: Write test that parses benchmark JSON
2. Implement: GitHub Actions workflow
3. Test: Trigger manually, verify artifact upload
4. Document: Add badge to README
5. Commit: `ci: Add automated model testing pipeline`

---

## Epic 7: Developer Experience & CLI Enhancements 🎨

**Business Value:** Reduce friction for new users, improve discoverability, make the CLI feel polished.

**User Stories:**
- As a new user, I want `brew install gemma` to work
- As a developer, I want tab completion for model IDs
- As a power user, I want to alias `gemma chat` to my preferred model

### 4.1 Installation & Distribution
**Status:** COMPLETED  
**Priority:** HIGH (for v0.2.0 release)  
**Effort:** 1 week

**Acceptance Criteria:**
- [x] Homebrew tap: `brew tap your-org/gemma && brew install gemma`
- [x] Binary releases on GitHub (macOS arm64)
- [x] Installation script: `curl -fsSL install.sh | bash`
- [x] Verify: `gemma --version` works after install

**Technical Design:**
```ruby
# Formula/gemma.rb
class Gemma < Formula
  desc "Local LLM inference server for Apple Silicon"
  homepage "https://github.com/your-org/GemmaServer"
  url "https://github.com/your-org/GemmaServer/archive/v0.2.0.tar.gz"
  sha256 "..."
  
  depends_on :macos
  depends_on arch: :arm64
  
  def install
    system "swift", "build", "-c", "release"
    bin.install ".build/release/GemmaServer" => "gemma"
  end
  
  test do
    assert_match "0.2.0", shell_output("#{bin}/gemma --version")
  end
end
```

**Workflow:**
1. Create: Homebrew tap repository
2. Implement: Formula + install script
3. Test: Fresh macOS VM, run install script
4. Document: Update README with install instructions
5. Commit: `chore: Add Homebrew formula and install script`

---

### 4.2 Interactive Model Selection
**Status:** COMPLETED  
**Priority:** MEDIUM  
**Effort:** 2 days

**Business Value:** Users don't need to memorize model IDs or browse HuggingFace.

**User Story:**
- As a user, I want `gemma chat` to show a menu if no model is specified
- As a user, I want to see RAM requirements before selecting a model

**Acceptance Criteria:**
- [x] `gemma chat` without `--model` shows interactive menu
- [x] Menu displays: model name, size, RAM, TPS, download status
- [x] Arrow keys to navigate, Enter to select
- [x] Auto-download if model not cached

**Technical Design:**
```swift
func interactiveModelPicker() async throws -> String {
    let models = [
        ("Qwen3.5-4B", "2.3 GB RAM, 92 TPS", "mlx-community/Qwen3.5-4B-4bit"),
        ("Qwen3.5-9B OptiQ", "5.8 GB RAM, 37 TPS", "mlx-community/Qwen3.5-9B-OptiQ-4bit"),
        ("Qwen2.5-Coder-7B", "4.1 GB RAM, 60 TPS", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
        ("Qwen3.6-27B", "14.5 GB RAM, 11 TPS", "mlx-community/Qwen3.6-27B-4bit"),
    ]
    
    print("Select a model:")
    for (i, (name, specs, _)) in models.enumerated() {
        let cached = ModelCache.isDownloaded(repoId: models[i].2) ? "✓" : " "
        print("  \(i+1). [\(cached)] \(name) — \(specs)")
    }
    
    print("\nEnter number (1-\(models.count)): ", terminator: "")
    guard let input = readLine(), let idx = Int(input), (1...models.count).contains(idx) else {
        throw CLIError.invalidSelection
    }
    
    return models[idx - 1].2
}
```

**Workflow:**
1. TDD: Write test for picker logic (mock stdin)
2. Implement: `interactiveModelPicker()` function
3. Integration: Add to `ChatCommand.run()`
4. Test: Manual testing with real terminal
5. Commit: `feat: Add interactive model selection menu`

---

### 4.3 Rich Terminal Output
**Status:** Not started  
**Priority:** LOW  
**Effort:** 3-4 days

**Business Value:** Professional appearance, easier to read streaming output.

**User Stories:**
- As a user, I want syntax highlighting for code blocks in chat
- As a developer, I want tables and lists to render properly
- As a power user, I want to disable colors for piping to files

**Acceptance Criteria:**
- [ ] Markdown rendering: **bold**, *italic*, `code`, ```blocks```
- [ ] Syntax highlighting for Swift, Python, JavaScript, Bash
- [ ] Tables rendered with box-drawing characters
- [ ] `--no-color` flag to disable ANSI codes

**Technical Design:**
- Use `swift-markdown` for parsing
- Use `Splash` for syntax highlighting
- Stream tokens, buffer until block boundary, then render

**Workflow:**
1. TDD: Write test for markdown → ANSI conversion
2. Implement: `MarkdownRenderer` class
3. Integration: Wrap `ChatCommand` output
4. Test: Verify with various markdown inputs
5. Commit: `feat: Add rich markdown rendering to chat`

---

## Epic 8: Security & Dependency Audit 🔒 ✅ (COMPLETED)

**Business Value:** Ensure GemmaServer is secure, compliant, and free from known vulnerabilities. Protect user data and prevent supply chain attacks.

**Completion Date:** April 28, 2025  
**Security Score:** 10/10 🏆  
**Status:** ✅ PASSED

**User Stories:**
- As a user, I want assurance that my data never leaves my machine ✅
- As a security engineer, I need to verify no dependencies have known CVEs ✅
- As a compliance officer, I need audit logs of all data access ✅
- As a developer, I want automated security checks in CI/CD ✅

### 5.1 Dependency Security Audit
**Status:** ✅ COMPLETED  
**Priority:** CRITICAL (before v1.0.0)  
**Effort:** 1 week (Completed in 1 day)

**Business Value:** Prevent supply chain attacks and known vulnerabilities from entering production.

**Acceptance Criteria:**
- [x] Audit all Swift Package Manager dependencies (37 packages scanned)
- [x] Check for known CVEs in all dependencies (0 critical found)
- [x] Verify dependency signatures and checksums
- [x] Document security posture of each dependency (SECURITY_AUDIT.md)
- [x] Set up automated dependency scanning in CI (.github/workflows/security-audit.yml)
- [x] Create dependency update policy (documented in SECURITY.md)

**Results:**
* ✅ 37 dependencies scanned via OSV API
* ✅ 0 CRITICAL vulnerabilities
* ✅ 0 HIGH severity issues
* ⚠️ 1 CVE in swift-crypto 4.0.0+ (NOT AFFECTING - we use 3.15.1)
* ✅ All dependencies from trusted sources (Apple, verified orgs)

**Dependencies Audited:**
```
ml-explore/mlx-swift (0.31.3) ✅
ml-explore/mlx-swift-lm (3.31.3) ✅
hummingbird-project/hummingbird (2.22.0) ✅
hummingbird-project/hummingbird-auth (2.1.0) ✅
stephencelis/SQLite.swift (0.16.0) ✅
vapor/jwt-kit (4.13.5) ✅
apple/swift-crypto (3.15.1) ✅
apple/swift-argument-parser (1.7.1) ✅
```

**Deliverables Created:**
- ✅ SECURITY_AUDIT.md (16 KB full report)
- ✅ scripts/security_audit.sh (automated scanning)
- ✅ .github/workflows/security-audit.yml (CI/CD integration)
- ✅ Updated SECURITY.md with best practices
```

**Audit Checklist:**
- [ ] No dependencies with HIGH/CRITICAL CVEs
- [ ] All dependencies from trusted sources (Apple, verified orgs)
- [ ] No transitive dependencies with known issues
- [ ] All dependencies actively maintained (commits in last 6 months)
- [ ] License compatibility verified (MIT, Apache 2.0, BSD)
- [ ] No dependencies with telemetry or phone-home behavior

**Workflow:**
1. Run: `swift package show-dependencies` — map full dependency tree
2. Audit: Check each dependency against CVE databases
3. Document: Create `SECURITY.md` with findings
4. Fix: Update or replace vulnerable dependencies
5. Automate: Add to CI/CD pipeline
6. Commit: `security: Complete dependency audit and update vulnerable packages`

---

### 5.2 CWE Security Audit (Common Weakness Enumeration)
**Status:** ✅ COMPLETED  
**Priority:** CRITICAL (before v1.0.0)  
**Effort:** 2 weeks (Completed in 1 day)

**Business Value:** Identify and fix common security weaknesses before they become exploits.

**User Stories:**
- As a security researcher, I want to verify GemmaServer follows secure coding practices ✅
- As a user, I need confidence that my API keys and data are protected ✅
- As a pentester, I want to see evidence of security testing ✅

**CWE Categories Audited:**

**1. CWE-20: Improper Input Validation**
- [x] Validate all user inputs (prompts, maxTokens, temperature) ✅
- [x] Sanitize file paths to prevent directory traversal ✅
- [x] Validate model IDs to prevent command injection ✅
- [x] Check JWT token format before parsing ✅

**Status:** ✅ PASS - All inputs validated at API boundary

**2. CWE-89: SQL Injection**
- [x] Use parameterized queries for all SQLite operations ✅
- [x] Never concatenate user input into SQL strings ✅
- [x] Audit all database queries in auth module ✅

**Status:** ✅ PASS - All queries use SQLite.swift parameterized API

**3. CWE-79: Cross-Site Scripting (XSS)**
- [x] Sanitize all output in REST API responses ✅
- [x] Escape HTML/JavaScript in error messages ✅
- [x] Use Content-Security-Policy headers ✅

**Status:** ✅ N/A - REST API only, no HTML rendering

**4. CWE-200: Information Exposure**
- [x] Never log passwords, API keys, or tokens ✅
- [x] Redact sensitive data in error messages ✅
- [x] Remove stack traces from production responses ✅
- [x] Audit all logger.error() calls ✅

**Status:** ✅ PASS - Sensitive data properly redacted

**5. CWE-259: Hard-coded Credentials**
- [x] No hard-coded passwords in source code ✅ **FIXED**
- [x] No default API keys ✅ **FIXED**
- [x] Force password change on first login ✅
- [x] Audit for "admin"/"password" defaults ✅ **REMOVED**

**Status:** ✅ FIXED - Removed "gemma-super-secret-key" and default admin

**6. CWE-327: Weak Cryptography**
- [x] Passwords hashed with bcrypt (not SHA256) ✅ **UPGRADED**
- [x] JWT tokens signed with strong keys (256-bit minimum) ✅
- [x] No MD5 or SHA1 for security purposes ✅
- [x] Use AES-256 for encryption ✅

**Status:** ✅ FIXED - Upgraded SHA256 → Bcrypt (cost=12)

**7. CWE-502: Unsafe Deserialization**
- [x] Type-safe Codable for all JSON ✅
- [x] No binary deserialization ✅

**Status:** ✅ PASS - Type-safe Swift Codable only

**8. CWE-862: Missing Authorization**
- [x] JWT authentication required for /api/* ✅
- [ ] Role-Based Access Control (RBAC) ⚠️ IN PROGRESS
- [ ] Admin-only routes protected ⚠️ PENDING

**Status:** ⚠️ PARTIAL - Basic auth works, RBAC needed for v1.0.0

**9. CWE-400: Uncontrolled Resource Consumption**
    guard (1...65_536).contains(request.maxTokens) else {
        throw .invalidInput("maxTokens must be 1-65536")
    }
    guard (0.0...2.0).contains(request.temperature) else {
        throw .invalidInput("temperature must be 0.0-2.0")
    }
}
```

**2. CWE-89: SQL Injection**
- [ ] Use parameterized queries for all SQLite operations
- [ ] Never concatenate user input into SQL strings
- [ ] Audit all database queries in auth module

```swift
// ✅ GOOD: Parameterized query
let stmt = try db.prepare("SELECT * FROM users WHERE username = ?")
let user = try stmt.run(username)

// ❌ BAD: SQL injection vulnerable
let query = "SELECT * FROM users WHERE username = '\(username)'"
```

**3. CWE-79: Cross-Site Scripting (XSS)**
- [ ] Sanitize all output in REST API responses
- [ ] Escape HTML/JavaScript in error messages
- [ ] Use Content-Security-Policy headers

**4. CWE-200: Information Exposure**
- [ ] Never log passwords, API keys, or tokens
- [ ] Redact sensitive data in error messages
- [ ] Remove stack traces from production responses
- [ ] Audit all logger.error() calls

```swift
// ✅ GOOD: Redacted logging
logger.error("Authentication failed", metadata: [
    "username": "\(username)",
    "reason": "invalid_credentials"
    // ❌ DON'T log: "password": "\(password)"
])
```

**5. CWE-259: Hard-coded Credentials**
- [ ] No hard-coded passwords in source code
- [ ] No default API keys
- [ ] Force password change on first login
- [ ] Audit for "admin"/"password" defaults

**6. CWE-311: Missing Encryption**
- [ ] Passwords hashed with bcrypt/Argon2 (not SHA256)
- [ ] JWT tokens signed with strong keys (256-bit minimum)
- [ ] HTTPS enforced for REST API (no HTTP fallback)
- [ ] Sensitive data encrypted at rest

```swift
// ✅ GOOD: Strong password hashing
import Crypto

func hashPassword(_ password: String) -> String {
    let salt = Data(UUID().uuidString.utf8)
    let hash = SHA256.hash(data: Data(password.utf8) + salt)
    return "\(salt.base64EncodedString()):\(hash.hexString)"
}

// TODO: Upgrade to Argon2 for production
```

**7. CWE-327: Weak Cryptography**
- [ ] No MD5 or SHA1 for security purposes
- [ ] Use SHA256+ for hashing
- [ ] Use AES-256 for encryption
- [ ] JWT signed with HS256 or RS256 (not none)

**8. CWE-352: Cross-Site Request Forgery (CSRF)**
- [ ] CSRF tokens for state-changing operations
- [ ] Validate Origin/Referer headers
- [ ] SameSite cookie attribute

**9. CWE-400: Uncontrolled Resource Consumption**
- [ ] Rate limiting on API endpoints
- [ ] Max request size limits
- [ ] Timeout for long-running operations
- [ ] Memory limits for model loading

```swift
// ✅ GOOD: Rate limiting
actor RateLimiter {
    private var requests: [String: [Date]] = [:]
    
    func checkLimit(clientId: String, maxRequests: Int, window: TimeInterval) throws {
        let now = Date()
        let windowStart = now.addingTimeInterval(-window)
        
        requests[clientId] = (requests[clientId] ?? [])
            .filter { $0 > windowStart }
        
        guard requests[clientId]!.count < maxRequests else {
            throw .rateLimitExceeded
        }
        
        requests[clientId]!.append(now)
    }
}
```

**10. CWE-502: Deserialization of Untrusted Data**
- [ ] Validate JSON structure before decoding
- [ ] Use Codable (type-safe) instead of JSONSerialization
- [ ] Never deserialize arbitrary objects

**Technical Design:**
```swift
// Security audit checklist implementation
struct SecurityAudit {
    func auditInputValidation() async throws -> [Finding]
    func auditSQLInjection() async throws -> [Finding]
    func auditInformationExposure() async throws -> [Finding]
    func auditCryptography() async throws -> [Finding]
    func auditResourceLimits() async throws -> [Finding]
    
    struct Finding {
        let cwe: String
        let severity: Severity
        let location: String
        let description: String
        let remediation: String
    }
    
    enum Severity {
        case critical, high, medium, low, info
    }
}
```

**Test Plan:**
1. Static analysis: Run SwiftLint security rules
2. Manual code review: Audit all security-sensitive code
3. Penetration testing: Attempt common exploits
4. Fuzzing: Test with malformed inputs
5. Dependency scan: Check for vulnerable libraries

**Workflow:**
1. Audit: Review all code against CWE checklist
2. Document: Create `SECURITY.md` with findings
3. Fix: Address all CRITICAL and HIGH findings
4. Test: Verify fixes with security tests
5. Automate: Add security checks to CI/CD
6. Commit: `security: Complete CWE audit and fix vulnerabilities`

---

### 5.3 Data Privacy & Compliance Audit
**Status:** Not started  
**Priority:** HIGH (before App Store)  
**Effort:** 1 week

**Business Value:** Ensure user data never leaves the device, comply with privacy regulations (GDPR, CCPA).

**Acceptance Criteria:**
- [ ] Audit all network calls — verify no telemetry
- [ ] Document data flows (what data goes where)
- [ ] Verify models run 100% locally (no cloud API calls)
- [ ] Create privacy policy
- [ ] Add privacy manifest (Apple requirement)

**Data Flow Audit:**
```
User Input (prompt) → ModelOrchestrator → MLX Inference → Response
                                ↓
                         Local cache only
                         NO network calls
```

**Network Call Audit:**
- [ ] HuggingFace downloads: Only for model files (user-initiated)
- [ ] No analytics/telemetry
- [ ] No crash reporting to external services
- [ ] No update checks (user-initiated only)

**Privacy Manifest (Apple App Store requirement):**
```json
{
  "NSPrivacyTracking": false,
  "NSPrivacyTrackingDomains": [],
  "NSPrivacyCollectedDataTypes": [],
  "NSPrivacyAccessedAPITypes": [
    {
      "NSPrivacyAccessedAPIType": "NSPrivacyAccessedAPICategoryFileTimestamp",
      "NSPrivacyAccessedAPITypeReasons": ["C617.1"]
    }
  ]
}
```

**Workflow:**
1. Audit: Review all URLSession calls
2. Document: Create data flow diagram
3. Test: Network monitoring during inference
4. Legal: Draft privacy policy
5. Commit: `docs: Add privacy policy and data flow documentation`

---

## Epic 9: MCP Plugin Marketplace & Agent Integration 🔌

**Business Value:** Enable GemmaServer to discover and integrate with MCP servers, creating an ecosystem of AI agents that can collaborate through standardized protocols.

**User Stories:**
- As a developer, I want to browse and install MCP servers from a marketplace
- As an agent builder, I want GemmaServer to auto-discover agent capabilities from `.md` files
- As a team, we need agents to communicate through standardized MCP protocol
- As a user, I want to connect Claude Desktop, Gemini agents, and custom skills seamlessly

### 6.1 MCP Server Discovery & Registry
**Status:** Not started  
**Priority:** HIGH (for ecosystem growth)  
**Effort:** 2-3 weeks

**Business Value:** Centralized registry of MCP servers enables discoverability and reduces integration friction.

**Acceptance Criteria:**
- [ ] Local registry: `~/.gemmaserver/mcp-registry.json`
- [ ] CLI command: `gemma mcp list` — show installed servers
- [ ] CLI command: `gemma mcp search <query>` — search public registry
- [ ] CLI command: `gemma mcp install <name>` — install from registry
- [ ] Auto-discovery: scan `~/.config/mcp/` for existing servers
- [ ] Parse MCP server manifests (JSON schema)

**Technical Design:**
```swift
struct MCPServerManifest: Codable, Sendable {
    let name: String
    let version: String
    let description: String
    let command: String
    let args: [String]
    let capabilities: [String]  // ["tools", "resources", "prompts"]
    let author: String?
    let repository: String?
}

actor MCPRegistry {
    private var servers: [String: MCPServerManifest] = [:]
    
    func register(manifest: MCPServerManifest) async throws {
        servers[manifest.name] = manifest
        try await persist()
    }
    
    func search(query: String) async -> [MCPServerManifest] {
        servers.values.filter { 
            $0.name.localizedCaseInsensitiveContains(query) ||
            $0.description.localizedCaseInsensitiveContains(query)
        }
    }
}
```

**Test Plan:**
1. Unit test: Register, search, uninstall operations
2. Integration test: Install real MCP server (e.g., filesystem, brave-search)
3. E2E test: GemmaServer connects to installed MCP server

**Workflow:**
1. TDD: Write test for `MCPRegistry` actor
2. Implement: Registry with JSON persistence
3. CLI: Add `mcp` subcommand group
4. Integration: Test with real MCP servers
5. Commit: `feat: Add MCP server registry and discovery`

---

### 6.2 Agent Capability Analysis (agents.md, gemini.md, claude-skill.md)
**Status:** COMPLETED  
**Priority:** HIGH  
**Effort:** 1-2 weeks

**Business Value:** Automatically extract agent capabilities from documentation files, enabling dynamic tool routing and agent collaboration.

**User Stories:**
- As a developer, I want GemmaServer to read `agents.md` and understand available tools
- As an agent, I need to discover what other agents can do without manual configuration
- As a user, I want seamless integration between Claude Desktop, Gemini, and custom agents

**Acceptance Criteria:**
- [x] Parser for `agents.md` (Anthropic format)
- [x] Parser for `gemini.md` (Google format)
- [x] Parser for `claude-skill.md` (custom skill definitions)
- [x] Extract: tool names, descriptions, parameters, return types
- [x] CLI command: `gemma agents analyze <file>` — show parsed capabilities- [ ] Auto-register tools in MCP server on startup

**Technical Design:**
```swift
struct AgentCapability: Codable, Sendable {
    let name: String
    let description: String
    let parameters: [Parameter]
    let returnType: String?
    let source: CapabilitySource
    
    enum CapabilitySource: String, Codable {
        case agentsMd = "agents.md"
        case geminiMd = "gemini.md"
        case claudeSkill = "claude-skill.md"
    }
}

actor AgentCapabilityAnalyzer {
    func parse(file: URL) async throws -> [AgentCapability] {
        let content = try String(contentsOf: file)
        let format = detectFormat(content)
        
        return switch format {
        case .agentsMd:   try parseAgentsMd(content)
        case .geminiMd:   try parseGeminiMd(content)
        case .claudeSkill: try parseClaudeSkill(content)
        }
    }
    
    private func parseAgentsMd(_ content: String) throws -> [AgentCapability] {
        // Parse Anthropic agents.md format:
        // ## Tool: <name>
        // Description: ...
        // Parameters: ...
    }
}
```

**Test Plan:**
1. Unit test: Parse sample `agents.md`, `gemini.md`, `claude-skill.md`
2. Integration test: Load capabilities into MCP server
3. E2E test: Agent calls tool discovered from `.md` file

**Workflow:**
1. TDD: Write tests with sample `.md` files
2. Implement: Markdown parser with regex/swift-markdown
3. CLI: Add `agents analyze` command
4. Integration: Auto-load on server startup
5. Commit: `feat: Add agent capability analysis from .md files`

---

### 6.3 MCP Plugin Marketplace UI
**Status:** Not started  
**Priority:** MEDIUM  
**Effort:** 3-4 weeks

**Business Value:** Web-based marketplace for browsing, installing, and managing MCP servers.

**User Stories:**
- As a user, I want a visual interface to browse available MCP servers
- As a developer, I want to publish my MCP server to the marketplace
- As a team, we need to share private MCP servers within our organization

**Acceptance Criteria:**
- [ ] Web UI: Browse marketplace (React/SwiftUI)
- [ ] Search, filter by category (filesystem, web, database, AI)
- [ ] Install button → downloads and registers MCP server
- [ ] Show installed servers with status (running/stopped)
- [ ] Start/stop/restart controls
- [ ] Logs viewer for each MCP server

**Technical Design:**
- Frontend: SwiftUI app (macOS) or React web app
- Backend: REST API endpoints for marketplace operations
- Database: SQLite for installed servers, status, logs

**Workflow:**
1. Design: Mockup UI in Figma
2. Backend: REST API for marketplace CRUD
3. Frontend: SwiftUI/React implementation
4. Integration: Connect to MCP registry
5. Commit: `feat: Add MCP plugin marketplace UI`

---

## Epic 10: Apple App Store Distribution 🍎

**Business Value:** Reach mainstream users through official Apple distribution, establish GemmaServer as a trusted local AI platform.

**User Stories:**
- As a non-technical user, I want to install GemmaServer from the Mac App Store
- As a developer, I want automatic updates through App Store
- As Apple, we need apps to follow sandboxing and entitlements guidelines

### 6.1 App Store Preparation
**Status:** Not started  
**Priority:** HIGH (for v1.0.0 release)  
**Effort:** 2-3 weeks

**Business Value:** App Store distribution increases trust, discoverability, and user base by 10-100x.

**Acceptance Criteria:**
- [ ] Create Apple Developer account ($99/year)
- [ ] App Store Connect: Create app listing
- [ ] App icon (1024x1024) in all required sizes
- [ ] Screenshots (macOS 13+, 14+, 15+)
- [ ] Privacy policy (data collection, model usage)
- [ ] Terms of service
- [ ] App description, keywords, category (Developer Tools)

**Workflow:**
1. Register: Apple Developer Program
2. Design: App icon + screenshots
3. Legal: Privacy policy + ToS
4. Submit: App Store Connect listing
5. Commit: `docs: Add App Store assets and legal docs`

---

### 6.2 Sandboxing & Entitlements
**Status:** Not started  
**Priority:** HIGH  
**Effort:** 1-2 weeks

**Business Value:** App Store requires sandboxing for security. Proper entitlements enable network, file access, and GPU usage.

**Acceptance Criteria:**
- [ ] Enable App Sandbox in Xcode
- [ ] Entitlements: `com.apple.security.network.server` (REST API)
- [ ] Entitlements: `com.apple.security.network.client` (HuggingFace downloads)
- [ ] Entitlements: `com.apple.security.files.user-selected.read-write` (model cache)
- [ ] Entitlements: `com.apple.security.device.metal` (GPU access)
- [ ] Test: All features work in sandboxed environment
- [ ] Hardened Runtime enabled

**Technical Design:**
```xml
<!-- GemmaServer.entitlements -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <key>com.apple.security.device.metal</key>
    <true/>
</dict>
</plist>
```

**Test Plan:**
1. Unit test: All tests pass with sandbox enabled
2. Integration test: Download model in sandbox
3. E2E test: Serve inference in sandbox
4. Manual test: Submit to App Store Review (TestFlight)

**Workflow:**
1. Enable: App Sandbox in Xcode project
2. Add: Required entitlements
3. Test: Full workflow in sandbox
4. Fix: Any sandbox violations
5. Commit: `feat: Add App Sandbox and entitlements for App Store`

---

### 6.3 Code Signing & Notarization
**Status:** Not started  
**Priority:** HIGH  
**Effort:** 3-5 days

**Business Value:** Code signing proves authenticity, notarization ensures malware-free distribution.

**Acceptance Criteria:**
- [ ] Developer ID Application certificate
- [ ] Sign all binaries with `codesign`
- [ ] Notarize app bundle with Apple
- [ ] Staple notarization ticket to app
- [ ] Verify: `spctl --assess --verbose` passes

**Technical Design:**
```bash
# Sign app bundle
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name (TEAM_ID)" \
  --options runtime \
  --entitlements GemmaServer.entitlements \
  GemmaServer.app

# Create ZIP for notarization
ditto -c -k --keepParent GemmaServer.app GemmaServer.zip

# Submit for notarization
xcrun notarytool submit GemmaServer.zip \
  --apple-id "your@email.com" \
  --team-id "TEAM_ID" \
  --password "app-specific-password" \
  --wait

# Staple ticket
xcrun stapler staple GemmaServer.app

# Verify
spctl --assess --verbose=4 --type execute GemmaServer.app
```

**Workflow:**
1. Generate: Developer ID certificate
2. Sign: All binaries
3. Notarize: Submit to Apple
4. Staple: Attach ticket
5. Commit: `chore: Add code signing and notarization`

---

### 6.4 App Store Submission & Review
**Status:** Not started  
**Priority:** HIGH  
**Effort:** 1-2 weeks (including review time)

**Business Value:** Final step to reach mainstream users.

**Acceptance Criteria:**
- [ ] Build archive in Xcode (Product → Archive)
- [ ] Upload to App Store Connect
- [ ] Fill out App Store Review Information
- [ ] Submit for review
- [ ] Respond to reviewer feedback (if any)
- [ ] App approved and live on App Store

**Workflow:**
1. Archive: Build release in Xcode
2. Upload: App Store Connect
3. Submit: For review
4. Monitor: Review status
5. Celebrate: App goes live 🎉

---

## Epic 10: Advanced Features (Future)
## Epic 11: Scheduled Tasks & Multi-Agent Orchestration ⏰

**Business Value:** Automated task execution with multiple AI agents (Gemini, Gemma, Qwen, Ollama, Copilot, Claude) enables complex workflows and 24/7 operation.

**User Stories:**
- As a developer, I want to schedule code reviews to run nightly
- As a team, we need different AI agents for different tasks (code vs docs vs architecture)
- As a user, I want agents to collaborate on complex tasks
- As an ops engineer, I need automated monitoring and reporting

### 11.1 Task Scheduler & Cron System
**Status:** Not started  
**Priority:** HIGH (for v0.4.0)  
**Effort:** 2 weeks

**Business Value:** Automated task execution reduces manual work and enables 24/7 AI assistance.

**Acceptance Criteria:**
- [ ] Cron-style task scheduling
- [ ] Task templates (code review, documentation, testing)
- [ ] Task history and logs
- [ ] Retry logic for failed tasks
- [ ] Email/Slack notifications on completion

**Technical Design:**
```swift
actor TaskScheduler {
    struct ScheduledTask: Codable, Sendable {
        let id: UUID
        let name: String
        let cronExpression: String  // "0 2 * * *" = 2 AM daily
        let agentType: AgentType
        let prompt: String
        let enabled: Bool
        let lastRun: Date?
        let nextRun: Date?
    }
    
    enum AgentType: String, Codable, Sendable {
        case gemma = "local-gemma"
        case qwen = "local-qwen"
        case ollama = "ollama-api"
        case gemini = "google-gemini"
        case claude = "anthropic-claude"
        case copilot = "github-copilot"
    }
    
    private var tasks: [UUID: ScheduledTask] = [:]
    private var timers: [UUID: Task<Void, Never>] = [:]
    
    func schedule(task: ScheduledTask) async throws {
        tasks[task.id] = task
        
        // Parse cron expression and calculate next run
        let nextRun = try calculateNextRun(cron: task.cronExpression)
        
        // Schedule task
        let timer = Task {
            while !Task.isCancelled {
                let delay = nextRun.timeIntervalSinceNow
                if delay > 0 {
                    try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
                
                if task.enabled {
                    await executeTask(task)
                }
                
                // Calculate next run
                let next = try? calculateNextRun(cron: task.cronExpression)
                guard let next else { break }
            }
        }
        
        timers[task.id] = timer
    }
    
    func executeTask(_ task: ScheduledTask) async {
        print("⏰ Executing scheduled task: \(task.name)")
        
        do {
            let agent = try await createAgent(type: task.agentType)
            let result = try await agent.execute(prompt: task.prompt)
            
            // Save result
            try await saveTaskResult(taskId: task.id, result: result)
            
            // Send notification
            await sendNotification(
                title: "Task Complete: \(task.name)",
                body: "Result: \(result.summary)"
            )
        } catch {
            print("❌ Task failed: \(error)")
            await sendNotification(
                title: "Task Failed: \(task.name)",
                body: "Error: \(error.localizedDescription)"
            )
        }
    }
}
```

**Cron Expression Parser:**
```swift
struct CronParser {
    // Format: minute hour day month weekday
    // Example: "0 2 * * *" = 2 AM every day
    
    func parse(_ expression: String) throws -> CronSchedule {
        let parts = expression.split(separator: " ")
        guard parts.count == 5 else {
            throw CronError.invalidFormat
        }
        
        return CronSchedule(
            minute: try parseField(String(parts[0]), range: 0...59),
            hour: try parseField(String(parts[1]), range: 0...23),
            day: try parseField(String(parts[2]), range: 1...31),
            month: try parseField(String(parts[3]), range: 1...12),
            weekday: try parseField(String(parts[4]), range: 0...6)
        )
    }
    
    func calculateNextRun(from date: Date, schedule: CronSchedule) -> Date {
        // Calculate next matching date/time
        var components = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: date)
        
        // Increment and find next match
        // ... implementation
        
        return Calendar.current.date(from: components)!
    }
}
```

**Workflow:**
1. Implement: CronParser and TaskScheduler
2. Add: Task persistence (SQLite)
3. Implement: Task execution engine
4. Test: Schedule tasks for various times
5. Commit: `feat: Add cron-style task scheduler`

---

### 11.2 Multi-Agent Orchestration
**Status:** Not started  
**Priority:** HIGH  
**Effort:** 2 weeks

**Business Value:** Different AI agents excel at different tasks. Orchestration enables best-of-breed approach.

**Acceptance Criteria:**
- [ ] Support 6 agent types (Gemma, Qwen, Ollama, Gemini, Claude, Copilot)
- [ ] Agent selection based on task type
- [ ] Sub-agent delegation (agent can call other agents)
- [ ] Agent collaboration (multiple agents on one task)
- [ ] Cost tracking (API calls for cloud agents)

**Agent Registry:**
```swift
actor AgentRegistry {
    enum AgentCapability {
        case codeGeneration
        case codeReview
        case documentation
        case architecture
        case testing
        case debugging
    }
    
    struct AgentProfile {
        let type: AgentType
        let capabilities: [AgentCapability]
        let costPerToken: Double?  // nil for local
        let maxTokens: Int
        let avgResponseTime: TimeInterval
    }
    
    let profiles: [AgentType: AgentProfile] = [
        .gemma: AgentProfile(
            type: .gemma,
            capabilities: [.codeGeneration, .codeReview, .documentation],
            costPerToken: nil,
            maxTokens: 128_000,
            avgResponseTime: 2.0
        ),
        .claude: AgentProfile(
            type: .claude,
            capabilities: [.codeGeneration, .codeReview, .architecture, .documentation],
            costPerToken: 0.000015,  // $15 per 1M tokens
            maxTokens: 200_000,
            avgResponseTime: 5.0
        ),
        .gemini: AgentProfile(
            type: .gemini,
            capabilities: [.codeGeneration, .testing, .debugging],
            costPerToken: 0.000001,  // $1 per 1M tokens
            maxTokens: 1_000_000,
            avgResponseTime: 3.0
        ),
        // ... other agents
    ]
    
    func selectAgent(for capability: AgentCapability, preferLocal: Bool = true) -> AgentType {
        let candidates = profiles.filter { $0.value.capabilities.contains(capability) }
        
        if preferLocal {
            // Prefer local agents (Gemma, Qwen)
            return candidates.first { $0.value.costPerToken == nil }?.key ?? .gemma
        } else {
            // Prefer fastest agent
            return candidates.min { $0.value.avgResponseTime < $1.value.avgResponseTime }?.key ?? .claude
        }
    }
}
```

**Agent Orchestrator:**
```swift
actor AgentOrchestrator {
    struct Task {
        let description: String
        let capability: AgentCapability
        let maxCost: Double?  // Budget limit
        let deadline: Date?
    }
    
    struct TaskResult {
        let agentUsed: AgentType
        let output: String
        let tokensUsed: Int
        let cost: Double
        let duration: TimeInterval
    }
    
    func execute(task: Task) async throws -> TaskResult {
        // Select best agent
        let registry = AgentRegistry()
        let agentType = await registry.selectAgent(
            for: task.capability,
            preferLocal: task.maxCost == nil || task.maxCost! < 0.01
        )
        
        // Create agent
        let agent = try await createAgent(type: agentType)
        
        // Execute
        let start = ContinuousClock.now
        let result = try await agent.execute(prompt: task.description)
        let duration = start.duration(to: .now).inSeconds
        
        // Calculate cost
        let profile = await registry.profiles[agentType]!
        let cost = Double(result.tokensUsed) * (profile.costPerToken ?? 0.0)
        
        return TaskResult(
            agentUsed: agentType,
            output: result.text,
            tokensUsed: result.tokensUsed,
            cost: cost,
            duration: duration
        )
    }
    
    // Sub-agent delegation
    func executeWithSubAgents(mainTask: Task, subTasks: [Task]) async throws -> [TaskResult] {
        // Execute sub-tasks in parallel
        try await withThrowingTaskGroup(of: TaskResult.self) { group in
            for subTask in subTasks {
                group.addTask {
                    try await self.execute(task: subTask)
                }
            }
            
            var results: [TaskResult] = []
            for try await result in group {
                results.append(result)
            }
            return results
        }
    }
}
```

**Example: Code Review with Multiple Agents:**
```swift
// Main task: Code review
let mainTask = Task(
    description: "Review this PR for bugs and style issues",
    capability: .codeReview,
    maxCost: 0.10,
    deadline: Date().addingTimeInterval(300)  // 5 minutes
)

// Sub-tasks
let subTasks = [
    Task(description: "Check for security vulnerabilities", capability: .debugging),
    Task(description: "Verify test coverage", capability: .testing),
    Task(description: "Review architecture changes", capability: .architecture)
]

let orchestrator = AgentOrchestrator()
let results = try await orchestrator.executeWithSubAgents(
    mainTask: mainTask,
    subTasks: subTasks
)

// Aggregate results
let summary = aggregateResults(results)
```

**Workflow:**
1. Implement: AgentRegistry with profiles
2. Implement: Agent adapters (Gemini API, Claude API, etc.)
3. Implement: AgentOrchestrator
4. Test: Execute tasks with different agents
5. Commit: `feat: Add multi-agent orchestration system`

---

### 11.3 Agent Collaboration & Consensus
**Status:** Not started  
**Priority:** MEDIUM  
**Effort:** 1 week

**Business Value:** Multiple agents can validate each other's work, reducing errors.

**Acceptance Criteria:**
- [ ] Multiple agents work on same task
- [ ] Consensus mechanism (voting, averaging)
- [ ] Conflict resolution
- [ ] Quality scoring

**Consensus Mechanism:**
```swift
actor ConsensusEngine {
    func executeWithConsensus(
        task: String,
        agents: [AgentType],
        consensusThreshold: Double = 0.7
    ) async throws -> String {
        // Execute task with multiple agents
        let results = try await withThrowingTaskGroup(of: (AgentType, String).self) { group in
            for agent in agents {
                group.addTask {
                    let a = try await createAgent(type: agent)
                    let result = try await a.execute(prompt: task)
                    return (agent, result.text)
                }
            }
            
            var outputs: [(AgentType, String)] = []
            for try await result in group {
                outputs.append(result)
            }
            return outputs
        }
        
        // Calculate similarity between outputs
        let similarities = calculateSimilarities(results.map(\.1))
        
        // Find consensus
        if similarities.max()! >= consensusThreshold {
            // High agreement - return most common answer
            return findMostCommon(results.map(\.1))
        } else {
            // Low agreement - ask meta-agent to resolve
            return try await resolveConflict(results: results)
        }
    }
    
    func resolveConflict(results: [(AgentType, String)]) async throws -> String {
        let metaPrompt = """
        Multiple AI agents provided different answers to the same question.
        Please analyze and provide the best answer:
        
        \(results.enumerated().map { "Agent \($0.offset + 1): \($0.element.1)" }.joined(separator: "\n\n"))
        """
        
        let metaAgent = try await createAgent(type: .claude)  // Use most capable
        let resolution = try await metaAgent.execute(prompt: metaPrompt)
        return resolution.text
    }
}
```

**Workflow:**
1. Implement: ConsensusEngine
2. Implement: Similarity calculation
3. Test: Run same task with 3 agents
4. Commit: `feat: Add agent collaboration and consensus`

---

## Epic 12: Telegram Bot Integration 📱

**Business Value:** Remote control of GemmaServer via Telegram enables mobile access and team collaboration.

**User Stories:**
- As a user, I want to start inference from my phone
- As a team, we need a shared Telegram bot for AI access
- As a developer, I want to monitor server status remotely
- As an admin, I need to manage scheduled tasks from mobile

### 12.1 Telegram Bot Setup & Authentication
**Status:** Not started  
**Priority:** MEDIUM (for v0.4.0)  
**Effort:** 1 week

**Business Value:** Mobile access to local AI without exposing REST API to internet.

**Acceptance Criteria:**
- [ ] Telegram bot registration
- [ ] User authentication (whitelist)
- [ ] Session management
- [ ] Command routing
- [ ] Rate limiting per user

**Technical Design:**
```swift
import TelegramBotSDK

actor TelegramBotServer {
    let bot: TelegramBot
    let orchestrator: ModelOrchestratorActor
    let allowedUsers: Set<Int64>  // Telegram user IDs
    
    init(token: String, orchestrator: ModelOrchestratorActor, allowedUsers: Set<Int64>) {
        self.bot = TelegramBot(token: token)
        self.orchestrator = orchestrator
        self.allowedUsers = allowedUsers
    }
    
    func start() async {
        // Register command handlers
        bot.onCommand("/start") { [weak self] update in
            await self?.handleStart(update)
        }
        
        bot.onCommand("/status") { [weak self] update in
            await self?.handleStatus(update)
        }
        
        bot.onCommand("/generate") { [weak self] update in
            await self?.handleGenerate(update)
        }
        
        bot.onCommand("/models") { [weak self] update in
            await self?.handleModels(update)
        }
        
        bot.onCommand("/schedule") { [weak self] update in
            await self?.handleSchedule(update)
        }
        
        // Start polling
        try await bot.run()
    }
    
    func handleStart(_ update: Update) async {
        guard let userId = update.message?.from?.id,
              allowedUsers.contains(userId) else {
            await bot.sendMessage(
                chatId: update.message!.chat.id,
                text: "❌ Unauthorized. Contact admin to get access."
            )
            return
        }
        
        await bot.sendMessage(
            chatId: update.message!.chat.id,
            text: """
            🤖 GemmaServer Bot
            
            Available commands:
            /status - Server status
            /generate <prompt> - Generate text
            /models - List models
            /schedule - Manage scheduled tasks
            /help - Show help
            """
        )
    }
    
    func handleGenerate(_ update: Update) async {
        guard let userId = update.message?.from?.id,
              allowedUsers.contains(userId) else {
            return
        }
        
        guard let text = update.message?.text,
              let prompt = text.split(separator: " ", maxSplits: 1).last else {
            await bot.sendMessage(
                chatId: update.message!.chat.id,
                text: "Usage: /generate <your prompt>"
            )
            return
        }
        
        // Send "typing" indicator
        await bot.sendChatAction(chatId: update.message!.chat.id, action: .typing)
        
        do {
            let request = GenerationRequest(prompt: String(prompt), maxTokens: 500)
            let response = try await orchestrator.generate(request: request)
            
            await bot.sendMessage(
                chatId: update.message!.chat.id,
                text: """
                ✨ Generated:
                
                \(response.generatedText)
                
                📊 Stats: \(response.tokensPerSecond) TPS, \(response.completionTokens) tokens
                """
            )
        } catch {
            await bot.sendMessage(
                chatId: update.message!.chat.id,
                text: "❌ Error: \(error.localizedDescription)"
            )
        }
    }
}
```

**Configuration:**
```swift
// Config file: ~/.gemmaserver/telegram.json
struct TelegramConfig: Codable {
    let botToken: String
    let allowedUsers: [Int64]
    let rateLimit: Int  // requests per minute
}
```

**Workflow:**
1. Register: Create Telegram bot via @BotFather
2. Implement: TelegramBotServer actor
3. Implement: Command handlers
4. Test: Send commands from Telegram app
5. Commit: `feat: Add Telegram bot integration`

---

### 12.2 Remote Session Management
**Status:** Not started  
**Priority:** MEDIUM  
**Effort:** 5 days

**Business Value:** Monitor and control GemmaServer from anywhere.

**Acceptance Criteria:**
- [ ] View server status (uptime, memory, active sessions)
- [ ] Start/stop inference sessions
- [ ] View scheduled tasks
- [ ] Cancel running tasks
- [ ] View logs

**Commands:**
```
/status - Show server status
/sessions - List active sessions
/tasks - List scheduled tasks
/cancel <task_id> - Cancel task
/logs - View recent logs
/restart - Restart server
```

**Status Command:**
```swift
func handleStatus(_ update: Update) async {
    let uptime = Date().timeIntervalSince(serverStartTime)
    let memory = Memory.snapshot()
    let activeSessions = await orchestrator.getActiveSessions()
    
    await bot.sendMessage(
        chatId: update.message!.chat.id,
        text: """
        📊 Server Status
        
        Uptime: \(formatDuration(uptime))
        Memory: \(memory.activeBytes / 1_048_576) MB
        Active Sessions: \(activeSessions.count)
        Model: \(currentModel ?? "None")
        
        Status: ✅ Running
        """
    )
}
```

**Workflow:**
1. Implement: Status monitoring
2. Implement: Session management commands
3. Implement: Task management commands
4. Test: Control server from Telegram
5. Commit: `feat: Add remote session management via Telegram`

---

### 12.3 Inline Queries & Notifications
**Status:** Not started  
**Priority:** LOW  
**Effort:** 3 days

**Business Value:** Quick access to AI without opening chat.

**Acceptance Criteria:**
- [ ] Inline queries (@bot <prompt>)
- [ ] Push notifications for task completion
- [ ] Keyboard shortcuts
- [ ] Rich formatting (Markdown)

**Inline Query Handler:**
```swift
bot.onInlineQuery { [weak self] query in
    guard let self = self else { return }
    
    let prompt = query.query
    guard !prompt.isEmpty else { return }
    
    // Generate quick response
    let request = GenerationRequest(prompt: prompt, maxTokens: 200)
    let response = try? await self.orchestrator.generate(request: request)
    
    let result = InlineQueryResultArticle(
        id: UUID().uuidString,
        title: "Generate with GemmaServer",
        inputMessageContent: InputTextMessageContent(
            messageText: response?.generatedText ?? "Error generating response"
        )
    )
    
    await bot.answerInlineQuery(inlineQueryId: query.id, results: [result])
}
```

**Workflow:**
1. Implement: Inline query handler
2. Implement: Push notifications
3. Test: Use inline queries
4. Commit: `feat: Add inline queries and notifications`

---

### 7.1 Streaming REST API
**Status:** ✅ COMPLETED  
**Business Value:** Real-time token streaming for web UIs  
**Effort:** 1 week (Completed in 1 day)  
**Priority:** MEDIUM

**Acceptance Criteria:**
- [x] Server-Sent Events (SSE) endpoint: `POST /api/v1/generate/stream` ✅
- [x] Chunked transfer encoding ✅
- [x] Client example (JavaScript fetch) ✅

**Completed Features:**
* ✅ SSE endpoint with proper headers (text/event-stream)
* ✅ StreamChunk: Codable conformance for JSON encoding
* ✅ Real-time token streaming via AsyncStream
* ✅ Error handling with event:error messages
* ✅ [DONE] completion message
* ✅ Interactive web demo (docs/examples/streaming-client.html)

**Technical Implementation:**
```swift
// GenerateController.generateStream()
let body = ResponseBody(asyncSequence: sseStream(dto: dto))
Response(status: .ok, headers: [
    .contentType: "text/event-stream",
    .cacheControl: "no-cache",
    .connection: "keep-alive"
], body: body)

### 7.2 Multi-Model Serving
**Business Value:** Switch models without restarting server  
**Effort:** 2 weeks  
**Priority:** LOW

- [ ] Load multiple models in parallel (if RAM allows)
- [ ] Request header: `X-Model-ID: qwen3.5-4b`
- [ ] LRU cache for model eviction

### 7.3 Quantization Experiments
**Business Value:** Research-grade quantization benchmarks  
**Effort:** 1 month (research project)  
**Priority:** LOW

- [ ] Integrate `mlx-lm` quantization tools
- [ ] Benchmark: 2-bit, 3-bit, 4-bit, 8-bit, FP16
- [ ] Quality metrics: perplexity, MMLU, HumanEval

---

## Development Workflow (Standard Operating Procedure)

### Testing Philosophy

**TDD (Test-Driven Development):**
- Write tests BEFORE implementation
- Red → Green → Refactor cycle
- Tests define the contract/interface
- Use for: Business logic, algorithms, data transformations

**BDD (Behavior-Driven Development):**
- Write tests in Given-When-Then format
- Focus on user behavior and outcomes
- Use for: API endpoints, user workflows, integration scenarios

**Test Coverage Requirements:**
- **100% unit test coverage** for all business logic
- **90%+ integration test coverage** for API endpoints
- **80%+ E2E test coverage** for critical user flows
- Exceptions: UI code, generated code, trivial getters/setters

**Coverage Enforcement:**
```bash
# Generate coverage report
swift test --enable-code-coverage

# Extract coverage percentage
xcrun llvm-cov report \
  .build/debug/GemmaServerPackageTests.xctest/Contents/MacOS/GemmaServerPackageTests \
  -instr-profile .build/debug/codecov/default.profdata \
  -use-color

# Fail CI if coverage < 100% for core modules
if [ $COVERAGE -lt 100 ]; then
  echo "ERROR: Unit test coverage is $COVERAGE%, required 100%"
  exit 1
fi
```

### For Each Task:

**1. TDD Phase (30 min - 1 hour)**
- **Write failing test FIRST** using Swift Testing
- Define interface (protocol/actor methods)
- Use mocks for external dependencies
- Run: `swift test` → verify test fails (RED)
- **Goal: 100% coverage of new code**

```swift
// Example: TDD for new feature
@Test("calculateMaxTokens returns correct limit for 16GB RAM")
func testCalculateMaxTokens_16GB() {
    let config = ServerConfig(availableRAM: 16_000_000_000, modelSizeMB: 4_000)
    let maxTokens = config.calculateMaxTokens()
    
    // Expected: (16GB * 0.8 - 4GB) / 2 bytes per token
    // = (12.8GB - 4GB) / 2 = 4.4GB / 2 = ~2.2B tokens
    // Capped at 128k
    #expect(maxTokens == 128_000)
}

// Run test → FAILS (method doesn't exist yet)
// Now implement calculateMaxTokens() to make it pass
```

**2. Implementation Phase (2-4 hours)**
- Write minimal code to pass test
- Follow Swift 6 concurrency rules
- Use pattern matching over if-else
- Run: `swift test` → verify test passes (GREEN)
- **Verify: Coverage remains 100%**

**3. Refactor Phase (30 min - 1 hour)**
- Improve code quality without changing behavior
- Extract duplicated code
- Simplify complex logic
- Run: `swift test` → all tests still pass (GREEN)
- **Verify: Coverage still 100%**

**4. BDD Integration Testing (1-2 hours)**
- Write Given-When-Then scenarios
- Test actor interactions
- Test concurrent access (50+ parallel requests)
- Test error propagation
- Run: `swift test` → all tests pass

```swift
// Example: BDD for API endpoint
@Test("POST /api/v1/generate returns response for valid request")
func testGenerateEndpoint_ValidRequest() async throws {
    // GIVEN: Server is running with loaded model
    let server = try await startTestServer()
    try await server.loadModel(path: testModelPath)
    
    // WHEN: Client sends valid generate request
    let request = GenerationRequest(prompt: "Hello", maxTokens: 100)
    let response = try await server.generate(request: request)
    
    // THEN: Response contains generated text and metrics
    #expect(!response.generatedText.isEmpty)
    #expect(response.tokensPerSecond > 0)
    #expect(response.promptTokens > 0)
}
```

**5. Profiling & Benchmarking (30 min - 1 hour)**
- Run `PerformanceBenchmark` before/after
- Measure TPS, TTFT, memory
- Compare: regression? → revert
- Document results in commit message

**6. Debug & Fix (as needed)**
- Use structured logging (not print)
- Check actor isolation
- Verify Sendable conformance
- Test edge cases

**6. Commit**
```bash
swift test  # All pass
git add Sources/ Tests/
git commit -m "feat: <description>

- Bullet point 1
- Bullet point 2

Benchmark results (Qwen3.5-4B):
  Before: 85 TPS
  After:  92 TPS (+8%)

Co-Authored-By: Claude Sonnet 4 <noreply@anthropic.com>"
```

**7. Update PLAN.md**
- Mark task as `[x]` completed
- Add notes if scope changed
- Update Epic status (IN PROGRESS → COMPLETED)
- **Update version counters** (see Version Tracking section)

**8. Version Tracking**
```bash
# After each commit, update version counters in PLAN.md

# Count commits by type
git log --oneline --grep="^feat:" | wc -l  # MINOR bumps
git log --oneline --grep="^fix:" | wc -l   # PATCH bumps
git log --oneline --grep="^BREAKING:" | wc -l  # MAJOR bumps

# Count completed features (tasks marked [x] in PLAN.md)
grep -c "\[x\]" PLAN.md

# Calculate next version
# If MAJOR changes: increment MAJOR, reset MINOR and PATCH
# If MINOR changes (feat:): increment MINOR, reset PATCH
# If PATCH changes (fix:): increment PATCH

# Update PLAN.md Version History section
# Update Feature Counter
# Update Commit Counter
```

**Version Bump Decision Tree:**
```
Commit message starts with:
├─ "BREAKING:" or "feat!:" → MAJOR bump (e.g., 1.0.0 → 2.0.0)
├─ "feat:" → MINOR bump (e.g., 0.1.0 → 0.2.0)
├─ "fix:" → PATCH bump (e.g., 0.1.0 → 0.1.1)
└─ "docs:", "chore:", "test:" → No bump

Epic completed:
└─ MINOR bump (e.g., 0.1.0 → 0.2.0)

Breaking API change:
└─ MAJOR bump (e.g., 0.9.0 → 1.0.0)
```

---

## Epic 13: Session Analytics & Usage Billing 💰

**Business Value:** Track usage, provide analytics, and enable usage-based billing for team deployments and commercial use.

**User Stories:**
- As a user, I want to see statistics about my inference sessions when I quit
- As a team lead, I need to track API usage across team members
- As a developer, I want to understand my token consumption patterns
- As a business, we need usage-based billing for enterprise customers

### 13.1 Session Analytics & Exit Summary
**Status:** Not started  
**Priority:** HIGH (for v0.4.0)  
**Effort:** 1 week

**Business Value:** Beautiful session summary on exit provides transparency and builds trust with users.

**Acceptance Criteria:**
- [ ] Track session metrics (start time, duration, requests, tokens)
- [ ] Beautiful ASCII art summary on `/quit` or SIGINT
- [ ] Display key statistics:
  - Session ID and duration (wall time vs active time)
  - Total requests and success rate
  - Token usage (input, output, total)
  - Model switches and cache hits
  - Performance metrics (avg TPS, TTFT)
  - Memory usage (peak, average)
- [ ] Color-coded output (green for good, yellow for warnings, red for errors)
- [ ] Export session summary to JSON

**Technical Design:**
```swift
actor SessionAnalytics {
    struct SessionMetrics: Codable, Sendable {
        let sessionId: UUID
        let startTime: Date
        let endTime: Date?
        
        var wallTime: TimeInterval { 
            (endTime ?? Date()).timeIntervalSince(startTime) 
        }
        
        var requests: [RequestMetric] = []
        var modelSwitches: Int = 0
        var cacheHits: Int = 0
        var cacheSize: Int64 = 0
        
        var totalInputTokens: Int {
            requests.map(\.inputTokens).reduce(0, +)
        }
        
        var totalOutputTokens: Int {
            requests.map(\.outputTokens).reduce(0, +)
        }
        
        var successRate: Double {
            guard !requests.isEmpty else { return 0 }
            let successful = requests.filter(\.success).count
            return Double(successful) / Double(requests.count) * 100
        }
        
        var avgTPS: Double {
            guard !requests.isEmpty else { return 0 }
            let totalTPS = requests.map(\.tokensPerSecond).reduce(0, +)
            return totalTPS / Double(requests.count)
        }
        
        var avgTTFT: Double {
            guard !requests.isEmpty else { return 0 }
            let totalTTFT = requests.map(\.timeToFirstToken).reduce(0, +)
            return totalTTFT / Double(requests.count)
        }
    }
    
    struct RequestMetric: Codable, Sendable {
        let timestamp: Date
        let duration: TimeInterval
        let inputTokens: Int
        let outputTokens: Int
        let tokensPerSecond: Double
        let timeToFirstToken: Double
        let success: Bool
        let error: String?
    }
    
    private var metrics = SessionMetrics(
        sessionId: UUID(),
        startTime: Date(),
        endTime: nil
    )
    
    func recordRequest(
        inputTokens: Int,
        outputTokens: Int,
        tps: Double,
        ttft: Double,
        duration: TimeInterval,
        success: Bool,
        error: String? = nil
    ) {
        let request = RequestMetric(
            timestamp: Date(),
            duration: duration,
            inputTokens: inputTokens,
            outputTokens: outputTokens,
            tokensPerSecond: tps,
            timeToFirstToken: ttft,
            success: success,
            error: error
        )
        metrics.requests.append(request)
    }
    
    func recordModelSwitch() {
        metrics.modelSwitches += 1
    }
    
    func recordCacheHit() {
        metrics.cacheHits += 1
    }
    
    func generateExitSummary() -> String {
        metrics.endTime = Date()
        
        let wallTime = formatDuration(metrics.wallTime)
        let successRate = String(format: "%.1f%%", metrics.successRate)
        let avgTPS = String(format: "%.1f", metrics.avgTPS)
        let avgTTFT = String(format: "%.3f", metrics.avgTTFT)
        
        return """
        ╭──────────────────────────────────────────────────────────────────────────────╮
        │                                                                              │
        │  GemmaServer powering down. Goodbye! 👋                                      │
        │                                                                              │
        │  Interaction Summary                                                         │
        │  Session ID:                 \(metrics.sessionId.uuidString.prefix(36))            │
        │  Inference Requests:         \(String(format: "%3d", metrics.requests.count)) ( ✓ \(metrics.requests.filter(\.success).count) ✗ \(metrics.requests.filter { !$0.success }.count) )                               │
        │  Success Rate:               \(successRate.padding(toLength: 6, withPad: " ", startingAt: 0))                                           │
        │  Model Switches:             \(metrics.modelSwitches)                                                │
        │  Cache Hits:                 \(metrics.cacheHits)                                                │
        │                                                                              │
        │  Performance                                                                 │
        │  Wall Time:                  \(wallTime.padding(toLength: 10, withPad: " ", startingAt: 0))                                      │
        │  Avg Tokens/sec:             \(avgTPS) TPS                                          │
        │  Avg Time to First Token:    \(avgTTFT)s                                           │
        │                                                                              │
        │  Token Usage                                                                 │
        │  Input Tokens:               \(String(format: "%,d", metrics.totalInputTokens).padding(toLength: 12, withPad: " ", startingAt: 0))                                    │
        │  Output Tokens:              \(String(format: "%,d", metrics.totalOutputTokens).padding(toLength: 12, withPad: " ", startingAt: 0))                                    │
        │  Total Tokens:               \(String(format: "%,d", metrics.totalInputTokens + metrics.totalOutputTokens).padding(toLength: 12, withPad: " ", startingAt: 0))                                    │
        │                                                                              │
        │  💡 Tip: Run `gemma stats --session \(metrics.sessionId)` to see detailed breakdown   │
        │                                                                              │
        ╰──────────────────────────────────────────────────────────────────────────────╯
        """
    }
    
    func exportToJSON() throws -> Data {
        metrics.endTime = Date()
        return try JSONEncoder().encode(metrics)
    }
}

func formatDuration(_ seconds: TimeInterval) -> String {
    let hours = Int(seconds) / 3600
    let minutes = (Int(seconds) % 3600) / 60
    let secs = Int(seconds) % 60
    
    if hours > 0 {
        return "\(hours)h \(minutes)m \(secs)s"
    } else if minutes > 0 {
        return "\(minutes)m \(secs)s"
    } else {
        return "\(secs)s"
    }
}
```

**Integration Points:**
```swift
// In ServeCommand.swift
class ServeCommand {
    let analytics = SessionAnalytics()
    
    func run() async throws {
        // Setup signal handlers for graceful shutdown
        signal(SIGINT) { _ in
            Task {
                await handleShutdown()
            }
        }
        
        // ... normal server operations
    }
    
    func handleShutdown() async {
        let summary = await analytics.generateExitSummary()
        print(summary)
        
        // Save to file
        if let data = try? await analytics.exportToJSON() {
            let url = URL(fileURLWithPath: "~/.gemmaserver/sessions/\(Date().ISO8601Format()).json")
            try? data.write(to: url)
        }
        
        exit(0)
    }
}

// In ModelOrchestratorActor.swift
func generate(request: GenerationRequest) async throws -> GenerationResponse {
    let start = ContinuousClock.now
    
    do {
        let response = try await engine.generate(request: request)
        let duration = start.duration(to: .now).inSeconds
        
        await analytics.recordRequest(
            inputTokens: response.promptTokens,
            outputTokens: response.completionTokens,
            tps: response.tokensPerSecond,
            ttft: response.timeToFirstToken,
            duration: duration,
            success: true
        )
        
        return response
    } catch {
        await analytics.recordRequest(
            inputTokens: 0,
            outputTokens: 0,
            tps: 0,
            ttft: 0,
            duration: start.duration(to: .now).inSeconds,
            success: false,
            error: error.localizedDescription
        )
        throw error
    }
}
```

**Workflow:**
1. Implement: SessionAnalytics actor
2. Integrate: Hook into ServeCommand shutdown
3. Test: Verify exit summary displays correctly
4. Polish: Add color coding with ANSI escape codes
5. Commit: `feat: Add session analytics and beautiful exit summary`

---

### 13.2 Usage Tracking & Billing Integration
**Status:** Not started  
**Priority:** MEDIUM (for v0.5.0)  
**Effort:** 2 weeks

**Business Value:** Enable usage-based billing for enterprise customers and SaaS deployments.

**Acceptance Criteria:**
- [ ] Track usage per user/API key
- [ ] Billable metrics: tokens, requests, compute time
- [ ] Export to billing systems (Stripe, Chargebee)
- [ ] Usage quotas and rate limiting per user
- [ ] Billing dashboard API endpoints
- [ ] Monthly usage reports

**Technical Design:**
```swift
actor UsageTracker {
    struct UserUsage: Codable {
        let userId: String
        var totalInputTokens: Int = 0
        var totalOutputTokens: Int = 0
        var totalRequests: Int = 0
        var totalComputeSeconds: Double = 0
        var billingPeriodStart: Date
        var billingPeriodEnd: Date
        
        var estimatedCost: Double {
            // Example pricing: $0.001 per 1K tokens
            let tokenCost = Double(totalInputTokens + totalOutputTokens) / 1000.0 * 0.001
            // Example compute: $0.01 per minute
            let computeCost = (totalComputeSeconds / 60.0) * 0.01
            return tokenCost + computeCost
        }
    }
    
    private var usage: [String: UserUsage] = [:]
    
    func recordUsage(userId: String, inputTokens: Int, outputTokens: Int, computeSeconds: Double) {
        var userUsage = usage[userId] ?? UserUsage(
            userId: userId,
            billingPeriodStart: Date(),
            billingPeriodEnd: Calendar.current.date(byAdding: .month, value: 1, to: Date())!
        )
        
        userUsage.totalInputTokens += inputTokens
        userUsage.totalOutputTokens += outputTokens
        userUsage.totalRequests += 1
        userUsage.totalComputeSeconds += computeSeconds
        
        usage[userId] = userUsage
    }
    
    func exportStripeInvoice(userId: String) throws -> StripeInvoice {
        guard let userUsage = usage[userId] else {
            throw BillingError.userNotFound
        }
        
        return StripeInvoice(
            customerId: userId,
            amount: Int(userUsage.estimatedCost * 100), // cents
            description: "GemmaServer Usage - \(userUsage.totalRequests) requests, \(userUsage.totalInputTokens + userUsage.totalOutputTokens) tokens"
        )
    }
}
```

**Workflow:**
1. Design: Usage tracking schema
2. Implement: UsageTracker actor
3. Integrate: Billing system APIs
4. Test: Verify accurate tracking
5. Commit: `feat: Add usage tracking and billing integration`

---

### 13.3 Analytics Dashboard
**Status:** Not started  
**Priority:** LOW  
**Effort:** 1 week

**Business Value:** Visualize usage patterns and optimize costs.

**Acceptance Criteria:**
- [ ] Web dashboard showing usage trends
- [ ] Charts: tokens over time, requests per hour, cost breakdown
- [ ] Filter by user, date range, model
- [ ] Export reports to CSV/PDF
- [ ] Real-time usage monitoring

**Workflow:**
1. Design: Dashboard mockups
2. Implement: REST API endpoints for analytics
3. Frontend: SwiftUI or React dashboard
4. Deploy: Serve dashboard from GemmaServer
5. Commit: `feat: Add usage analytics dashboard`

---

## Success Metrics

**v0.1.0 (Current):**
- ✅ 51/51 tests passing
- ✅ 4 verified models
- ✅ Dual interface (MCP + REST)
- ✅ JWT authentication
- ✅ 100% unit test coverage for core modules

**v0.2.0 (Next Release):**
- [x] Context degradation profiler
- [x] Dynamic token budgeting
- [x] Homebrew installation
- [ ] 10+ verified models
- [x] CI/CD pipeline
- [ ] Dependency security audit completed
- [ ] CWE security audit completed
- [ ] 100% unit test coverage maintained

**v1.0.0 (Production):**
- [ ] 99.9% uptime (no crashes)
- [ ] <100ms TTFT for 4B models
- [ ] Support all major model families (Gemma, Qwen, Llama, Mistral)
- [ ] 1000+ GitHub stars
- [ ] Live on Mac App Store
- [ ] MCP plugin marketplace with 10+ servers
- [x] Agent capability discovery from .md files
- [ ] Zero known HIGH/CRITICAL security vulnerabilities
- [ ] Privacy audit completed and documented
- [ ] GDPR/CCPA compliant
- [ ] 100% unit test coverage across all modules
- [ ] 90%+ integration test coverage
- [ ] 80%+ E2E test coverage

---

## Epic 14: Cloud Model Integration (OpenRouter API) ☁️

**Business Value:** Enable hybrid inference by routing requests to cloud models when local resources are insufficient or when accessing frontier models not available locally.

**User Stories:**
- As a developer, I want to seamlessly switch between local MLX models and cloud models via OpenRouter
- As a user with limited RAM, I need access to larger models (70B+) through cloud inference
- As a team lead, I want cost-effective access to GPT-4, Claude 3.5, and other frontier models
- As a researcher, I need to compare local vs cloud model performance side-by-side

**Strategic Rationale:**
- **Hybrid Architecture:** Combine local privacy (MLX) with cloud scale (OpenRouter)
- **Cost Optimization:** Use local models for development, cloud for production queries
- **Model Access:** Access 100+ models without downloading 100GB+ weights
- **Fallback Strategy:** Auto-failover from local to cloud when OOM occurs
- **A/B Testing:** Compare local quantized vs cloud full-precision models

### 14.1 OpenRouter Client Implementation
**Status:** Not started  
**Priority:** HIGH (for v0.5.0)  
**Effort:** 1 week

**Business Value:** Core HTTP client for OpenRouter API with authentication and error handling.

**Acceptance Criteria:**
- [ ] HTTP client with async/await (URLSession or AsyncHTTPClient)
- [ ] API key management (environment variable + keychain storage)
- [ ] Request/response DTOs compatible with OpenRouter spec
- [ ] Retry logic with exponential backoff
- [ ] Rate limiting and quota tracking
- [ ] Error mapping (API errors → GemmaServerError)
- [ ] Request/response logging (with PII redaction)
- [ ] Timeout configuration (default: 60s)

**Technical Design:**
```swift
// Sources/GemmaServer/Cloud/OpenRouterClient.swift

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// OpenRouter API client for cloud model inference
actor OpenRouterClient: Sendable {
    
    // MARK: - Configuration
    
    struct Config: Sendable {
        let apiKey: String
        let baseURL: URL
        let timeout: TimeInterval
        let maxRetries: Int
        
        static let `default` = Config(
            apiKey: ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"] ?? "",
            baseURL: URL(string: "https://openrouter.ai/api/v1")!,
            timeout: 60.0,
            maxRetries: 3
        )
    }
    
    // MARK: - DTOs (OpenAI-compatible)
    
    struct ChatRequest: Codable, Sendable {
        let model: String
        let messages: [Message]
        let temperature: Double?
        let maxTokens: Int?
        let stream: Bool?
        
        struct Message: Codable, Sendable {
            let role: String  // "system", "user", "assistant"
            let content: String
        }
        
        enum CodingKeys: String, CodingKey {
            case model
            case messages
            case temperature
            case maxTokens = "max_tokens"
            case stream
        }
    }
    
    struct ChatResponse: Codable, Sendable {
        let id: String
        let model: String
        let choices: [Choice]
        let usage: Usage?
        
        struct Choice: Codable, Sendable {
            let message: Message
            let finishReason: String?
            
            struct Message: Codable, Sendable {
                let role: String
                let content: String
            }
            
            enum CodingKeys: String, CodingKey {
                case message
                case finishReason = "finish_reason"
            }
        }
        
        struct Usage: Codable, Sendable {
            let promptTokens: Int
            let completionTokens: Int
            let totalTokens: Int
            
            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
            }
        }
    }
    
    // MARK: - Properties
    
    private let config: Config
    private let session: URLSession
    private var requestCount: Int = 0
    private var errorCount: Int = 0
    
    // MARK: - Initialization
    
    init(config: Config = .default) throws {
        guard !config.apiKey.isEmpty else {
            throw GemmaServerError.invalidRequestStructure(
                details: "OPENROUTER_API_KEY not set. Get your key at https://openrouter.ai/keys"
            )
        }
        
        self.config = config
        
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = config.timeout
        configuration.timeoutIntervalForResource = config.timeout * 2
        self.session = URLSession(configuration: configuration)
    }
    
    // MARK: - API Methods
    
    /// Generate completion from cloud model
    func chat(request: ChatRequest) async throws -> ChatResponse {
        let url = config.baseURL.appendingPathComponent("chat/completions")
        
        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.setValue("GemmaServer/0.2.0", forHTTPHeaderField: "HTTP-Referer")
        httpRequest.setValue("GemmaServer", forHTTPHeaderField: "X-Title")
        
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        httpRequest.httpBody = try encoder.encode(request)
        
        // Retry logic with exponential backoff
        var lastError: Error?
        for attempt in 0..<config.maxRetries {
            do {
                let (data, response) = try await session.data(for: httpRequest)
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw GemmaServerError.invalidRequestStructure(details: "Invalid response type")
                }
                
                // Handle HTTP errors
                guard (200...299).contains(httpResponse.statusCode) else {
                    let errorBody = String(data: data, encoding: .utf8) ?? "Unknown error"
                    
                    switch httpResponse.statusCode {
                    case 401:
                        throw GemmaServerError.authenticationFailed(
                            details: "Invalid OpenRouter API key"
                        )
                    case 429:
                        throw GemmaServerError.rateLimitExceeded(
                            retryAfter: httpResponse.value(forHTTPHeaderField: "Retry-After")
                        )
                    case 500...599:
                        // Retry on server errors
                        if attempt < config.maxRetries - 1 {
                            let delay = pow(2.0, Double(attempt)) // Exponential backoff
                            try await Task.sleep(for: .seconds(delay))
                            continue
                        }
                        throw GemmaServerError.modelInferenceError(
                            details: "OpenRouter API error: \(errorBody)"
                        )
                    default:
                        throw GemmaServerError.invalidRequestStructure(
                            details: "HTTP \(httpResponse.statusCode): \(errorBody)"
                        )
                    }
                }
                
                // Parse response
                let decoder = JSONDecoder()
                decoder.keyDecodingStrategy = .convertFromSnakeCase
                let chatResponse = try decoder.decode(ChatResponse.self, from: data)
                
                await recordSuccess()
                return chatResponse
                
            } catch let error as GemmaServerError {
                throw error
            } catch {
                lastError = error
                if attempt < config.maxRetries - 1 {
                    let delay = pow(2.0, Double(attempt))
                    try await Task.sleep(for: .seconds(delay))
                }
            }
        }
        
        await recordError()
        throw lastError ?? GemmaServerError.modelInferenceError(
            details: "OpenRouter API request failed after \(config.maxRetries) attempts"
        )
    }
    
    // MARK: - Metrics
    
    private func recordSuccess() {
        requestCount += 1
    }
    
    private func recordError() {
        errorCount += 1
    }
    
    func getMetrics() -> (requests: Int, errors: Int) {
        return (requestCount, errorCount)
    }
}
```

**Workflow:**
1. TDD: Write tests for API key validation, request/response parsing
2. Implement: OpenRouterClient actor with retry logic
3. Integration: Add to dependency injection container
4. Test: Mock OpenRouter responses, test error handling
5. Commit: `feat: Add OpenRouter API client`

---

### 14.2 Model Router & Selection Strategy
**Status:** Not started  
**Priority:** HIGH (for v0.5.0)  
**Effort:** 1 week

**Business Value:** Intelligent routing between local MLX models and cloud OpenRouter models based on request characteristics and system resources.

**Acceptance Criteria:**
- [ ] Model registry with local + cloud models
- [ ] Routing strategy: Auto, Local-only, Cloud-only, Hybrid
- [ ] Cost estimation before cloud API calls
- [ ] Automatic fallback: Local → Cloud on OOM
- [ ] Model alias mapping (e.g., "gpt-4" → "openai/gpt-4-turbo")
- [ ] Request routing based on:
  - Model availability (local vs cloud)
  - System RAM availability
  - Request context size
  - Cost constraints
  - Latency requirements

**Technical Design:**
```swift
// Sources/GemmaServer/Cloud/ModelRouter.swift

actor ModelRouter: Sendable {
    
    enum RoutingStrategy: String, Sendable, Codable {
        case auto           // Smart routing based on resources
        case localOnly      // Never use cloud (privacy-first)
        case cloudOnly      // Always use cloud (no local inference)
        case hybrid         // Explicitly choose per request
    }
    
    enum ModelLocation: Sendable {
        case local(path: String)
        case cloud(provider: CloudProvider, modelId: String)
        
        enum CloudProvider: String, Sendable {
            case openrouter
            case openai
            case anthropic
        }
    }
    
    struct ModelInfo: Sendable {
        let id: String
        let displayName: String
        let location: ModelLocation
        let contextWindow: Int
        let estimatedRAM: Int64?       // MB (for local models)
        let costPer1MTokens: Double?   // USD (for cloud models)
    }
    
    private let localOrchestrator: ModelOrchestratorActor
    private let cloudClient: OpenRouterClient?
    private let strategy: RoutingStrategy
    private var modelRegistry: [String: ModelInfo] = [:]
    
    init(
        localOrchestrator: ModelOrchestratorActor,
        cloudClient: OpenRouterClient?,
        strategy: RoutingStrategy = .auto
    ) {
        self.localOrchestrator = localOrchestrator
        self.cloudClient = cloudClient
        self.strategy = strategy
        
        // Pre-populate registry with known models
        registerKnownModels()
    }
    
    /// Route request to appropriate model (local or cloud)
    func route(request: GenerationRequest, preferredModel: String?) async throws -> GenerationResponse {
        let modelId = preferredModel ?? "auto"
        
        // Determine routing decision
        let decision = try await makeRoutingDecision(
            modelId: modelId,
            request: request
        )
        
        switch decision {
        case .local(let path):
            return try await routeToLocal(path: path, request: request)
            
        case .cloud(let provider, let cloudModelId):
            guard let client = cloudClient else {
                throw GemmaServerError.invalidRequestStructure(
                    details: "Cloud models not configured. Set OPENROUTER_API_KEY"
                )
            }
            return try await routeToCloud(
                client: client,
                modelId: cloudModelId,
                request: request
            )
        }
    }
    
    private func makeRoutingDecision(
        modelId: String,
        request: GenerationRequest
    ) async throws -> ModelLocation {
        
        // Strategy: Local-only
        if strategy == .localOnly {
            return .local(path: modelId)
        }
        
        // Strategy: Cloud-only
        if strategy == .cloudOnly {
            let cloudModel = mapToCloudModel(modelId)
            return .cloud(provider: .openrouter, modelId: cloudModel)
        }
        
        // Strategy: Auto or Hybrid
        // Check if model explicitly cloud-only (e.g., "gpt-4", "claude-3.5-sonnet")
        if isCloudOnlyModel(modelId) {
            let cloudModel = mapToCloudModel(modelId)
            return .cloud(provider: .openrouter, modelId: cloudModel)
        }
        
        // Check if local model available and sufficient RAM
        if let localInfo = modelRegistry[modelId],
           case .local = localInfo.location {
            
            let availableRAM = await getAvailableRAM()
            let requiredRAM = localInfo.estimatedRAM ?? 0
            
            // Sufficient RAM for local inference
            if availableRAM > requiredRAM * 1_024 * 1_024 * 2 { // 2x safety margin
                return .local(path: modelId)
            }
            
            // Insufficient RAM → fallback to cloud
            if strategy == .auto, cloudClient != nil {
                log("⚠️ Insufficient RAM for local model, falling back to cloud")
                let cloudModel = mapToCloudModel(modelId)
                return .cloud(provider: .openrouter, modelId: cloudModel)
            }
            
            // No cloud fallback available
            throw GemmaServerError.insufficientMemory(
                required: requiredRAM,
                available: Int(availableRAM / 1_024 / 1_024)
            )
        }
        
        // Unknown model → try cloud
        if cloudClient != nil {
            let cloudModel = mapToCloudModel(modelId)
            return .cloud(provider: .openrouter, modelId: cloudModel)
        }
        
        throw GemmaServerError.invalidRequestStructure(
            details: "Model '\(modelId)' not found in local or cloud registry"
        )
    }
    
    private func routeToLocal(path: String, request: GenerationRequest) async throws -> GenerationResponse {
        // Use existing local orchestrator
        return try await localOrchestrator.generate(request: request)
    }
    
    private func routeToCloud(
        client: OpenRouterClient,
        modelId: String,
        request: GenerationRequest
    ) async throws -> GenerationResponse {
        // Convert GemmaServer request to OpenRouter format
        let chatRequest = OpenRouterClient.ChatRequest(
            model: modelId,
            messages: [
                .init(role: "user", content: request.prompt)
            ],
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: false
        )
        
        let startTime = ContinuousClock.now
        let chatResponse = try await client.chat(request: chatRequest)
        let endTime = ContinuousClock.now
        
        let duration = startTime.duration(to: endTime).components.seconds
        
        // Convert OpenRouter response back to GemmaServer format
        guard let choice = chatResponse.choices.first else {
            throw GemmaServerError.modelInferenceError(
                details: "No response from cloud model"
            )
        }
        
        let usage = chatResponse.usage ?? .init(
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0
        )
        
        return GenerationResponse(
            generatedText: choice.message.content,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens,
            tokensPerSecond: Double(usage.completionTokens) / Double(duration),
            generationTime: Double(duration),
            timeToFirstToken: 0.0, // Not available from API
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
            finishReason: choice.finishReason == "stop" ? .stop : .length
        )
    }
    
    // MARK: - Helpers
    
    private func isCloudOnlyModel(_ modelId: String) -> Bool {
        let cloudPrefixes = ["gpt-", "claude-", "gemini-", "o1-", "o3-"]
        return cloudPrefixes.contains { modelId.hasPrefix($0) }
    }
    
    private func mapToCloudModel(_ modelId: String) -> String {
        // Map friendly names to OpenRouter model IDs
        let mapping: [String: String] = [
            "gpt-4": "openai/gpt-4-turbo",
            "gpt-4o": "openai/gpt-4o",
            "claude": "anthropic/claude-3.5-sonnet",
            "claude-3.5": "anthropic/claude-3.5-sonnet",
            "gemini": "google/gemini-pro-1.5",
            "o1": "openai/o1-preview",
            "o3": "openai/o3-mini"
        ]
        
        return mapping[modelId] ?? modelId
    }
    
    private func registerKnownModels() {
        // Register cloud models
        modelRegistry["gpt-4"] = ModelInfo(
            id: "gpt-4",
            displayName: "GPT-4 Turbo",
            location: .cloud(provider: .openrouter, modelId: "openai/gpt-4-turbo"),
            contextWindow: 128_000,
            estimatedRAM: nil,
            costPer1MTokens: 10.0
        )
        
        modelRegistry["claude-3.5"] = ModelInfo(
            id: "claude-3.5",
            displayName: "Claude 3.5 Sonnet",
            location: .cloud(provider: .openrouter, modelId: "anthropic/claude-3.5-sonnet"),
            contextWindow: 200_000,
            estimatedRAM: nil,
            costPer1MTokens: 3.0
        )
        
        // Local models registered dynamically
    }
    
    private func getAvailableRAM() async -> Int64 {
        // Use SystemProfiler or similar
        return 16 * 1024 * 1024 * 1024 // Placeholder
    }
    
    private func log(_ message: String) {
        fputs("[ModelRouter] \(message)\n", stderr)
    }
}
```

**Workflow:**
1. TDD: Write tests for routing logic (local vs cloud decisions)
2. Implement: ModelRouter with decision tree
3. Integration: Wire into REST/MCP handlers
4. Test: Simulate OOM scenarios, verify fallback
5. Commit: `feat: Add model router with cloud fallback`

---

### 14.3 Cost Tracking & Budget Limits
**Status:** Not started  
**Priority:** MEDIUM (for v0.5.0)  
**Effort:** 3 days

**Business Value:** Prevent unexpected cloud API bills by tracking usage and enforcing budget limits.

**Acceptance Criteria:**
- [ ] Per-session cost tracking
- [ ] Daily/monthly budget limits
- [ ] Cost estimation before request
- [ ] Warning at 80% budget consumption
- [ ] Hard stop at 100% budget
- [ ] Cost breakdown by model
- [ ] Export cost report (CSV/JSON)

**Technical Design:**
```swift
actor CostTracker: Sendable {
    struct Budget: Codable, Sendable {
        let dailyLimit: Double    // USD
        let monthlyLimit: Double  // USD
    }
    
    struct Usage: Codable, Sendable {
        var totalCost: Double = 0.0
        var requestCount: Int = 0
        var tokenCount: Int = 0
        var costByModel: [String: Double] = [:]
    }
    
    private var dailyUsage: Usage = .init()
    private var monthlyUsage: Usage = .init()
    private let budget: Budget
    
    func recordRequest(
        model: String,
        promptTokens: Int,
        completionTokens: Int
    ) async throws {
        let cost = calculateCost(
            model: model,
            promptTokens: promptTokens,
            completionTokens: completionTokens
        )
        
        // Check budget before charging
        if dailyUsage.totalCost + cost > budget.dailyLimit {
            throw GemmaServerError.budgetExceeded(
                limit: budget.dailyLimit,
                current: dailyUsage.totalCost
            )
        }
        
        dailyUsage.totalCost += cost
        dailyUsage.requestCount += 1
        dailyUsage.tokenCount += promptTokens + completionTokens
        dailyUsage.costByModel[model, default: 0.0] += cost
        
        // Warning at 80%
        if dailyUsage.totalCost > budget.dailyLimit * 0.8 {
            log("⚠️ Warning: 80% of daily budget consumed (\(dailyUsage.totalCost)/\(budget.dailyLimit) USD)")
        }
    }
    
    private func calculateCost(
        model: String,
        promptTokens: Int,
        completionTokens: Int
    ) -> Double {
        // OpenRouter pricing (approximate)
        let pricing: [String: (input: Double, output: Double)] = [
            "openai/gpt-4-turbo": (0.01, 0.03),        // per 1K tokens
            "anthropic/claude-3.5-sonnet": (0.003, 0.015),
            "google/gemini-pro-1.5": (0.00125, 0.005)
        ]
        
        guard let price = pricing[model] else {
            return 0.0 // Unknown model
        }
        
        let inputCost = Double(promptTokens) / 1000.0 * price.input
        let outputCost = Double(completionTokens) / 1000.0 * price.output
        
        return inputCost + outputCost
    }
}
```

---

### 14.4 Streaming Support for Cloud Models
**Status:** Not started  
**Priority:** MEDIUM (for v0.5.0)  
**Effort:** 1 week

**Business Value:** Real-time token streaming from cloud models for better UX.

**Acceptance Criteria:**
- [ ] SSE streaming from OpenRouter
- [ ] AsyncStream<StreamChunk> interface
- [ ] Unified streaming API (local + cloud)
- [ ] Error handling mid-stream
- [ ] Cancel support (stop generation)

---

### 14.5 CLI & Configuration
**Status:** Not started  
**Priority:** HIGH (for v0.5.0)  
**Effort:** 2 days

**Business Value:** Easy configuration and testing of cloud integration.

**Acceptance Criteria:**
- [ ] `GemmaServer cloud configure` - setup API key
- [ ] `GemmaServer cloud test` - verify connection
- [ ] `GemmaServer cloud models` - list available models
- [ ] `GemmaServer cloud cost` - show usage/cost
- [ ] Environment variable: `OPENROUTER_API_KEY`
- [ ] Config file: `~/.gemmaserver/cloud.json`

```bash
# Setup
export OPENROUTER_API_KEY="sk-or-v1-..."
GemmaServer cloud configure --budget-daily 10.00 --budget-monthly 100.00

# Test connection
GemmaServer cloud test

# List available models
GemmaServer cloud models --filter frontier

# Generate with cloud model
GemmaServer chat --model gpt-4 --prompt "Explain quantum computing"

# Check costs
GemmaServer cloud cost --period today
```

---

### 14.6 Documentation & Examples
**Status:** Not started  
**Priority:** MEDIUM (for v0.5.0)  
**Effort:** 2 days

**Acceptance Criteria:**
- [ ] docs/cloud/OPENROUTER_SETUP.md
- [ ] docs/cloud/COST_OPTIMIZATION.md
- [ ] docs/cloud/MODEL_COMPARISON.md
- [ ] Example: Hybrid inference (local dev, cloud prod)
- [ ] Example: Cost-aware routing
- [ ] Security best practices (API key storage)

---

## Summary: Epic 14 Task Breakdown

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| 14.1 OpenRouter Client | HIGH | 1 week | OpenRouterClient.swift |
| 14.2 Model Router | HIGH | 1 week | ModelRouter.swift |
| 14.3 Cost Tracking | MEDIUM | 3 days | CostTracker.swift |
| 14.4 Streaming Support | MEDIUM | 1 week | Cloud SSE streaming |
| 14.5 CLI Configuration | HIGH | 2 days | `cloud` subcommands |
| 14.6 Documentation | MEDIUM | 2 days | Setup guides |

**Total Effort:** ~4 weeks (1 month)  
**Target Version:** v0.5.0  
**Dependencies:** Epic 7.1 (Streaming API), Epic 8 (Security)

**Risk Mitigation:**
- API key security: Use Keychain, never commit to git
- Cost control: Hard limits enforced in code
- Privacy: Local-first default, explicit opt-in for cloud
- Reliability: Retry logic + fallback to local
