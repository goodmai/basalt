# Security & Dependency Audit Report

## Dependency Audit

An automated Open Source Vulnerability (OSV) check was performed on the `GemmaServer` dependencies. 

### Findings
- **swift-crypto**: A HIGH severity vulnerability (CVE-2026-28815) affects `swift-crypto` versions 4.0.0 through 4.3.0. The vulnerability involves an out-of-bounds read in the C decapsulation path for X-Wing HPKE encapsulated keys.
- **Remediation**: GemmaServer specifies `.package(url: "https://github.com/apple/swift-crypto.git", from: "3.0.0")` in `Package.swift`. This restricts the package to the `3.x` branch, avoiding the vulnerable `4.x` versions. No changes to the dependency tree were strictly necessary, but maintaining this constraint ensures security.

## CWE Security Audit

The following Common Weakness Enumeration (CWE) categories were audited and addressed:

### 1. CWE-311 / CWE-327: Missing Encryption / Weak Cryptography
- **Status**: Remedied.
- **Details**: Previously, user passwords (including the default `admin` account) were hashed using `SHA256` with a salt. This is vulnerable to dictionary and brute-force attacks due to high hashing speeds.
- **Fix**: Replaced `SHA256` with `Bcrypt` (using `vapor/bcrypt`) for robust, computationally-expensive password hashing. A seamless migration logic was added to gracefully upgrade old SHA256 hashes to Bcrypt upon the next successful login.

### 2. CWE-400: Uncontrolled Resource Consumption (Denial of Service)
- **Status**: Remedied.
- **Details**: The REST API endpoints (such as `/api/v1/generate`) were vulnerable to resource exhaustion from flood attacks.
- **Fix**: Implemented `RateLimitMiddleware` to restrict requests per remote IP address (configured to 50 requests per minute). Requests exceeding this limit return `HTTP 429 Too Many Requests`.

### 3. CWE-20: Improper Input Validation
- **Status**: Verified.
- **Details**: `GenerationRequest` strictly validates prompt lengths, requested `maxTokens` (bounding to max context lengths), and `temperature` variables to prevent injection and unexpected application state.

### 4. CWE-89: SQL Injection
- **Status**: Verified.
- **Details**: The project uses `SQLite.swift`, which natively utilizes parameterized queries for all database interactions. No raw string interpolation is performed on user input.

## Data Privacy
- **Status**: Verified.
- **Details**: GemmaServer performs 100% of its inference locally on-device. No telemetry, analytics, or background data collection is performed. Network requests are exclusively restricted to downloading model weights from the Hugging Face Hub (user-initiated). See `docs/PRIVACY_POLICY.md` and `PrivacyInfo.xcprivacy` for explicit privacy declarations.
