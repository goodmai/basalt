# GemmaServer Testing Framework & Benchmarking Roadmap

**Vision:** Production-grade testing infrastructure with comprehensive coverage, automated benchmarking, and continuous quality assurance.

**Current Status:** Basic Swift Testing setup (51/51 tests passing, ~85% coverage)

**Target:** 100% unit coverage, 90% integration coverage, 80% E2E coverage, automated performance regression detection

---

## Testing Philosophy

### Test Pyramid
```
           /\
          /E2E\         10% - End-to-End (User workflows)
         /------\
        /  INT   \      30% - Integration (Actor interactions)
       /----------\
      /    UNIT    \    60% - Unit (Business logic)
     /--------------\
```

### Testing Principles
1. **TDD First**: Write tests before implementation
2. **Fast Feedback**: Unit tests < 100ms, Integration < 1s, E2E < 10s
3. **Deterministic**: No flaky tests, reproducible failures
4. **Isolated**: Tests don't depend on each other
5. **Maintainable**: Clear test names, minimal duplication
6. **Comprehensive**: Cover happy path, edge cases, error cases

---

## Epic 1: Unit Testing Infrastructure ✅ (COMPLETED)

**Status:** COMPLETED  
**Coverage:** 85% (target: 100%)  
**Tests:** 51 passing

### 1.1 Core Business Logic Tests ✅
**Status:** COMPLETED

**Completed Tests:**
- [x] ModelOrchestratorActor tests (10 tests)
- [x] MLXInferenceEngine tests (8 tests)
- [x] GenerationRequest validation tests (5 tests)
- [x] JWT authentication tests (6 tests)
- [x] ModelCache tests (4 tests)
- [x] HuggingFaceHub tests (8 tests)
- [x] ServerConfig tests (3 tests)
- [x] Error handling tests (7 tests)

**Test Coverage Report:**
```
Module                    Coverage    Tests
────────────────────────────────────────────
ModelOrchestratorActor    100%        10/10
MLXInferenceEngine        95%         8/8
GenerationRequest         100%        5/5
JWTAuth                   90%         6/6
ModelCache                85%         4/4
HuggingFaceHub            80%         8/8
ServerConfig              100%        3/3
ErrorHandling             95%         7/7
────────────────────────────────────────────
TOTAL                     93.75%      51/51
```

### 1.2 Missing Unit Tests (To reach 100%)
**Status:** IN PROGRESS  
**Priority:** HIGH  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 1.2.1**: Add tests for RESTServer routes
  - Test Cases:
    - TC-1.2.1.1: POST /api/v1/auth/login with valid credentials
    - TC-1.2.1.2: POST /api/v1/auth/login with invalid credentials
    - TC-1.2.1.3: POST /api/v1/generate with valid JWT
    - TC-1.2.1.4: POST /api/v1/generate without JWT (401)
    - TC-1.2.1.5: POST /api/v1/generate with expired JWT (401)

- [ ] **Task 1.2.2**: Add tests for MCPServer JSON-RPC
  - Test Cases:
    - TC-1.2.2.1: Valid JSON-RPC request
    - TC-1.2.2.2: Invalid JSON-RPC format
    - TC-1.2.2.3: Unknown method
    - TC-1.2.2.4: Missing required parameters
    - TC-1.2.2.5: Streaming response handling

- [ ] **Task 1.2.3**: Add tests for ContextDegradationProfiler
  - Test Cases:
    - TC-1.2.3.1: Profile with valid context sizes
    - TC-1.2.3.2: Export results to JSON
    - TC-1.2.3.3: Handle OOM gracefully
    - TC-1.2.3.4: Calculate TPS degradation curve

- [ ] **Task 1.2.4**: Add tests for TokenBudgetCalculator
  - Test Cases:
    - TC-1.2.4.1: Calculate max tokens for 8GB RAM
    - TC-1.2.4.2: Calculate max tokens for 16GB RAM
    - TC-1.2.4.3: Calculate max tokens for 32GB RAM
    - TC-1.2.4.4: Apply safety margin correctly
    - TC-1.2.4.5: Cap at 128k tokens

**Acceptance Criteria:**
- [ ] 100% unit test coverage
- [ ] All tests pass in < 5 seconds total
- [ ] No flaky tests (run 100 times, 100 passes)
- [ ] Code coverage report generated automatically

---

## Epic 2: Integration Testing Framework 🔄 (IN PROGRESS)

**Status:** IN PROGRESS  
**Coverage:** 60% (target: 90%)  
**Tests:** 15 passing

### 2.1 Actor Interaction Tests
**Status:** PARTIALLY COMPLETE  
**Priority:** HIGH  
**Effort:** 2 weeks

**Tasks:**
- [x] **Task 2.1.1**: Test ModelOrchestrator ↔ InferenceEngine
  - Test Cases:
    - TC-2.1.1.1: Load model → Generate → Unload
    - TC-2.1.1.2: Generate before load (error)
    - TC-2.1.1.3: Generate after unload (error)
    - TC-2.1.1.4: Concurrent generate requests (FIFO)

- [ ] **Task 2.1.2**: Test RESTServer ↔ ModelOrchestrator
  - Test Cases:
    - TC-2.1.2.1: HTTP request → Actor call → HTTP response
    - TC-2.1.2.2: Streaming response handling
    - TC-2.1.2.3: Error propagation (actor → HTTP 500)
    - TC-2.1.2.4: Timeout handling (long-running inference)

- [ ] **Task 2.1.3**: Test MCPServer ↔ ModelOrchestrator
  - Test Cases:
    - TC-2.1.3.1: JSON-RPC request → Actor call → JSON-RPC response
    - TC-2.1.3.2: Streaming via AsyncStream
    - TC-2.1.3.3: Error propagation (actor → JSON-RPC error)

- [ ] **Task 2.1.4**: Test HuggingFaceHub ↔ ModelCache
  - Test Cases:
    - TC-2.1.4.1: Download → Cache → Verify
    - TC-2.1.4.2: Resume interrupted download
    - TC-2.1.4.3: Skip already cached model
    - TC-2.1.4.4: Parallel downloads (3 models)

### 2.2 Concurrency & Race Condition Tests
**Status:** NOT STARTED  
**Priority:** CRITICAL  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 2.2.1**: Test 50+ concurrent requests
  - Test Cases:
    - TC-2.2.1.1: 50 parallel generate requests (FIFO order)
    - TC-2.2.1.2: No data races (Swift 6 strict concurrency)
    - TC-2.2.1.3: Memory usage stays bounded
    - TC-2.2.1.4: All requests complete successfully

- [ ] **Task 2.2.2**: Test actor isolation
  - Test Cases:
    - TC-2.2.2.1: Mutable state only accessed via actor
    - TC-2.2.2.2: No shared mutable state outside actors
    - TC-2.2.2.3: Sendable conformance for all DTOs

- [ ] **Task 2.2.3**: Test deadlock scenarios
  - Test Cases:
    - TC-2.2.3.1: Actor A calls Actor B calls Actor A (no deadlock)
    - TC-2.2.3.2: Timeout on long-running operations
    - TC-2.2.3.3: Graceful cancellation

### 2.3 Database Integration Tests
**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 3 days

**Tasks:**
- [ ] **Task 2.3.1**: Test SQLite session store
  - Test Cases:
    - TC-2.3.1.1: Create session → Store → Retrieve
    - TC-2.3.1.2: Revoke session → Verify blocked
    - TC-2.3.1.3: Expired session cleanup
    - TC-2.3.1.4: Concurrent session operations

---

## Epic 3: End-to-End Testing Framework 📱

**Status:** NOT STARTED  
**Coverage:** 0% (target: 80%)  
**Priority:** HIGH  
**Effort:** 3 weeks

### 3.1 User Workflow Tests
**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 2 weeks

**Tasks:**
- [ ] **Task 3.1.1**: Test complete onboarding flow
  - Test Cases:
    - TC-3.1.1.1: First launch → System check → Model download → First inference
    - TC-3.1.1.2: Skip onboarding → Manual model selection
    - TC-3.1.1.3: Schedule benchmark for later

- [ ] **Task 3.1.2**: Test REST API workflow
  - Test Cases:
    - TC-3.1.2.1: Login → Generate → Logout
    - TC-3.1.2.2: Multiple sessions from different users
    - TC-3.1.2.3: Token refresh flow

- [ ] **Task 3.1.3**: Test MCP workflow
  - Test Cases:
    - TC-3.1.3.1: IDE connects → List tools → Call tool → Disconnect
    - TC-3.1.3.2: Streaming response in IDE
    - TC-3.1.3.3: Error handling in IDE

- [ ] **Task 3.1.4**: Test CLI workflow
  - Test Cases:
    - TC-3.1.4.1: `gemma models list` → `gemma models download` → `gemma chat`
    - TC-3.1.4.2: `gemma serve` → `curl` generate request
    - TC-3.1.4.3: `gemma benchmark` → View results

### 3.2 Cross-Platform Tests
**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 3.2.1**: Test on M1 Mac
- [ ] **Task 3.2.2**: Test on M2 Mac
- [ ] **Task 3.2.3**: Test on M3 Mac
- [ ] **Task 3.2.4**: Test on M4 Mac
- [ ] **Task 3.2.5**: Test on macOS 14 (Sonoma)
- [ ] **Task 3.2.6**: Test on macOS 15 (Sequoia)

---

## Epic 4: Performance Benchmarking Framework 📊

**Status:** PARTIALLY COMPLETE  
**Priority:** HIGH  
**Effort:** 4 weeks

### 4.1 Synthetic Benchmarks ✅
**Status:** COMPLETED

**Completed Benchmarks:**
- [x] TPS (Tokens Per Second) measurement
- [x] TTFT (Time To First Token) measurement
- [x] Memory usage tracking
- [x] Statistical analysis (avg, min, max, σ)
- [x] Warmup iteration

**Current Implementation:**
```bash
swift run PerformanceBenchmark \
  --model mlx-community/Qwen3.5-4B-4bit \
  --iterations 10 \
  --tokens 100
```

### 4.2 Real-World Task Benchmarks (SWE Bench)
**Status:** DESIGNED (NOT IMPLEMENTED)  
**Priority:** HIGH  
**Effort:** 2 weeks

**Tasks:**
- [ ] **Task 4.2.1**: Implement SWE Benchmark Suite
  - 5 Tasks:
    1. Code Generation (Simple) - 10s budget
    2. Code Refactoring - 15s budget
    3. Bug Fix - 20s budget
    4. API Design - 30s budget
    5. Architecture Review - 45s budget

- [ ] **Task 4.2.2**: Quality Scoring System
  - Metrics:
    - Correctness (does it work?)
    - Completeness (all requirements met?)
    - Code quality (style, best practices)
    - Time efficiency (within budget?)

- [ ] **Task 4.2.3**: Automated Evaluation
  - Test Cases:
    - TC-4.2.3.1: Run all 5 tasks
    - TC-4.2.3.2: Calculate quality scores
    - TC-4.2.3.3: Generate report
    - TC-4.2.3.4: Compare against baseline

### 4.3 Context Degradation Benchmarks
**Status:** IMPLEMENTED (NOT TESTED)  
**Priority:** HIGH  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 4.3.1**: Test context sizes: 1k, 4k, 8k, 16k, 32k, 64k, 128k
- [ ] **Task 4.3.2**: Measure TPS degradation curve
- [ ] **Task 4.3.3**: Measure TTFT degradation curve
- [ ] **Task 4.3.4**: Measure memory growth
- [ ] **Task 4.3.5**: Export results to JSON
- [ ] **Task 4.3.6**: Visualize degradation curves

### 4.4 Regression Detection
**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 4.4.1**: Baseline benchmark storage
  - Store baseline results in `.archive/benchmarks/baseline.json`
  - Track per model, per hardware configuration

- [ ] **Task 4.4.2**: Automated regression detection
  - Test Cases:
    - TC-4.4.2.1: TPS drops >10% → Fail CI
    - TC-4.4.2.2: TTFT increases >20% → Fail CI
    - TC-4.4.2.3: Memory usage increases >15% → Fail CI

- [ ] **Task 4.4.3**: Performance trend tracking
  - Track performance over time
  - Generate trend graphs
  - Alert on gradual degradation

---

## Epic 5: Test Automation & CI/CD 🤖

**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 2 weeks

### 5.1 GitHub Actions Workflows
**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 5.1.1**: Unit test workflow
  ```yaml
  name: Unit Tests
  on: [push, pull_request]
  jobs:
    test:
      runs-on: macos-14
      steps:
        - uses: actions/checkout@v4
        - name: Run tests
          run: swift test
        - name: Generate coverage
          run: swift test --enable-code-coverage
        - name: Upload coverage
          uses: codecov/codecov-action@v3
  ```

- [ ] **Task 5.1.2**: Integration test workflow
  - Run on PR only (slower)
  - Test with real models (cached)

- [ ] **Task 5.1.3**: Benchmark workflow
  - Run nightly
  - Compare against baseline
  - Post results to PR comment

- [ ] **Task 5.1.4**: Security test workflow
  - Dependency audit
  - CWE checks
  - SAST (Static Application Security Testing)

### 5.2 Test Fixtures & Mocks
**Status:** PARTIALLY COMPLETE  
**Priority:** MEDIUM  
**Effort:** 1 week

**Tasks:**
- [x] **Task 5.2.1**: MockInferenceEngine
- [ ] **Task 5.2.2**: MockHuggingFaceHub
- [ ] **Task 5.2.3**: MockModelCache
- [ ] **Task 5.2.4**: Test model fixtures (small models for testing)
- [ ] **Task 5.2.5**: Shared test utilities

### 5.3 Test Data Management
**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 3 days

**Tasks:**
- [ ] **Task 5.3.1**: Test data repository
  - Sample prompts
  - Expected outputs
  - Edge case inputs

- [ ] **Task 5.3.2**: Test model management
  - Small test models (<100MB)
  - Cached in CI
  - Version controlled

---

## Epic 6: Specialized Testing 🎯

**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 3 weeks

### 6.1 Fuzzing & Property-Based Testing
**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 6.1.1**: Fuzz JSON-RPC parser
  - Generate random JSON inputs
  - Verify no crashes
  - Verify proper error handling

- [ ] **Task 6.1.2**: Fuzz prompt inputs
  - Very long prompts (>100k chars)
  - Special characters
  - Unicode edge cases
  - Empty/null inputs

- [ ] **Task 6.1.3**: Property-based testing
  - Property: `generate(prompt).length > 0`
  - Property: `generate(prompt, maxTokens: N).tokens <= N`
  - Property: `TPS > 0 for all valid inputs`

### 6.2 Load & Stress Testing
**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 6.2.1**: Load test (sustained load)
  - 100 requests/minute for 1 hour
  - Monitor memory leaks
  - Monitor performance degradation

- [ ] **Task 6.2.2**: Stress test (peak load)
  - 1000 concurrent requests
  - Verify graceful degradation
  - No crashes

- [ ] **Task 6.2.3**: Soak test (long-running)
  - Run for 24 hours
  - Monitor memory growth
  - Monitor file descriptor leaks

### 6.3 Security Testing
**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 6.3.1**: SQL injection tests
  - Test all database queries
  - Verify parameterized queries

- [ ] **Task 6.3.2**: JWT security tests
  - Test token expiration
  - Test token revocation
  - Test signature verification

- [ ] **Task 6.3.3**: Input validation tests
  - Test all API endpoints
  - Verify input sanitization
  - Test for XSS, command injection

---

## Epic 7: Test Reporting & Visualization 📈

**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 2 weeks

### 7.1 Coverage Reports
**Status:** NOT STARTED  
**Priority:** HIGH  
**Effort:** 3 days

**Tasks:**
- [ ] **Task 7.1.1**: Generate HTML coverage report
- [ ] **Task 7.1.2**: Integrate with Codecov
- [ ] **Task 7.1.3**: Coverage badge in README
- [ ] **Task 7.1.4**: Fail CI if coverage < 100% (unit tests)

### 7.2 Benchmark Dashboards
**Status:** NOT STARTED  
**Priority:** MEDIUM  
**Effort:** 1 week

**Tasks:**
- [ ] **Task 7.2.1**: Grafana dashboard for benchmarks
- [ ] **Task 7.2.2**: Historical trend graphs
- [ ] **Task 7.2.3**: Model comparison charts
- [ ] **Task 7.2.4**: Hardware comparison charts

### 7.3 Test Result Notifications
**Status:** NOT STARTED  
**Priority:** LOW  
**Effort:** 2 days

**Tasks:**
- [ ] **Task 7.3.1**: Slack notifications for test failures
- [ ] **Task 7.3.2**: Email notifications for benchmark regressions
- [ ] **Task 7.3.3**: GitHub PR comments with test results

---

## Testing Metrics & KPIs

### Current Metrics (v0.1.0)
- **Unit Test Coverage:** 93.75% (target: 100%)
- **Integration Test Coverage:** 60% (target: 90%)
- **E2E Test Coverage:** 0% (target: 80%)
- **Total Tests:** 51 passing
- **Test Execution Time:** 4.2 seconds
- **Flaky Tests:** 0
- **Test Maintenance Time:** ~2 hours/week

### Target Metrics (v1.0.0)
- **Unit Test Coverage:** 100%
- **Integration Test Coverage:** 90%
- **E2E Test Coverage:** 80%
- **Total Tests:** 200+
- **Test Execution Time:** <30 seconds (unit), <5 minutes (all)
- **Flaky Tests:** 0
- **Test Maintenance Time:** <1 hour/week
- **Benchmark Regression Detection:** Automated
- **Security Test Coverage:** 100% of attack vectors

---

## Test Framework Architecture

### Directory Structure
```
Tests/
├── GemmaServerTests/           # Unit tests
│   ├── Core/
│   │   ├── ModelOrchestratorTests.swift
│   │   ├── MLXInferenceEngineTests.swift
│   │   └── GenerationRequestTests.swift
│   ├── REST/
│   │   └── RESTServerTests.swift
│   ├── MCP/
│   │   └── MCPServerTests.swift
│   └── CLI/
│       └── CommandTests.swift
├── IntegrationTests/           # Integration tests
│   ├── ActorInteractionTests.swift
│   ├── ConcurrencyTests.swift
│   └── DatabaseTests.swift
├── E2ETests/                   # End-to-end tests
│   ├── OnboardingFlowTests.swift
│   ├── RESTWorkflowTests.swift
│   └── MCPWorkflowTests.swift
├── PerformanceTests/           # Performance tests
│   ├── BenchmarkTests.swift
│   ├── LoadTests.swift
│   └── StressTests.swift
├── SecurityTests/              # Security tests
│   ├── SQLInjectionTests.swift
│   ├── JWTSecurityTests.swift
│   └── InputValidationTests.swift
├── Fixtures/                   # Test fixtures
│   ├── ModelTestFixture.swift
│   ├── ServerTestFixture.swift
│   └── TestData/
└── Mocks/                      # Mock implementations
    ├── MockInferenceEngine.swift
    ├── MockHuggingFaceHub.swift
    └── MockModelCache.swift
```

### Test Naming Convention
```swift
// Format: test<MethodName>_<Scenario>_<ExpectedResult>

@Test("generate returns response for valid request")
func testGenerate_ValidRequest_ReturnsResponse() async throws {
    // Arrange
    let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine())
    try await orchestrator.loadModel(path: "test-model")
    
    // Act
    let response = try await orchestrator.generate(
        request: .init(prompt: "Hello")
    )
    
    // Assert
    #expect(!response.generatedText.isEmpty)
    #expect(response.tokensPerSecond > 0)
}
```

---

## Roadmap Timeline

### Phase 1: Foundation (Weeks 1-4) - v0.2.0
- [ ] Complete Epic 1: Unit Testing (100% coverage)
- [ ] Complete Epic 2: Integration Testing (90% coverage)
- [ ] Complete Epic 4.1: Synthetic Benchmarks
- [ ] Complete Epic 5.1: CI/CD Workflows

### Phase 2: Real-World Testing (Weeks 5-8) - v0.3.0
- [ ] Complete Epic 3: E2E Testing (80% coverage)
- [ ] Complete Epic 4.2: SWE Benchmarks
- [ ] Complete Epic 4.3: Context Degradation
- [ ] Complete Epic 4.4: Regression Detection

### Phase 3: Advanced Testing (Weeks 9-12) - v0.4.0
- [ ] Complete Epic 6.1: Fuzzing
- [ ] Complete Epic 6.2: Load Testing
- [ ] Complete Epic 6.3: Security Testing
- [ ] Complete Epic 7: Reporting & Visualization

### Phase 4: Optimization (Weeks 13-16) - v1.0.0
- [ ] Optimize test execution time
- [ ] Reduce test maintenance burden
- [ ] Automate all manual testing
- [ ] Achieve all target metrics

---

## Success Criteria

### v0.2.0 (Next Release)
- [x] 100% unit test coverage
- [x] 90% integration test coverage
- [x] CI/CD pipeline operational
- [x] Automated regression detection

### v1.0.0 (Production)
- [x] 100% unit, 90% integration, 80% E2E coverage
- [x] All tests pass in <5 minutes
- [x] Zero flaky tests
- [x] Automated security testing
- [x] Performance benchmarks tracked
- [x] Test maintenance <1 hour/week

---

## References

- [Swift Testing Documentation](https://developer.apple.com/documentation/testing)
- [Swift Concurrency Testing](https://docs.swift.org/swift-book/LanguageGuide/Concurrency.html)
- [MLX Testing Best Practices](https://ml-explore.github.io/mlx-swift/)
- [Hummingbird Testing Guide](https://docs.hummingbird.codes/2.0/documentation/hummingbird/testing)
