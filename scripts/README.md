# GemmaServer Scripts

Automation scripts for development workflow.

## Version Management

### `update_version.sh`

Automatically calculates the next version based on commit history and epic completion.

**Usage:**
```bash
./scripts/update_version.sh
```

**What it does:**
1. Analyzes git commit history
2. Counts commits by type (feat, fix, BREAKING, docs, chore, security)
3. Counts completed features and epics
4. Calculates next version according to Semantic Versioning
5. Optionally updates PLAN.md with new counters

**Version Bump Rules:**
- `feat:` commits → MINOR bump (0.1.0 → 0.2.0)
- `fix:` commits → PATCH bump (0.1.0 → 0.1.1)
- `BREAKING:` or `feat!:` → MAJOR bump (0.9.0 → 1.0.0)
- `docs:`, `chore:`, `test:` → No bump
- Epic completion → MINOR bump

**Example Output:**
```
=== GemmaServer Version Counter ===

Current Version: v0.1.0
  MAJOR: 0
  MINOR: 1
  PATCH: 0

Commit Analysis:
  feat:     12 commits (MINOR bumps)
  fix:      2 commits (PATCH bumps)
  BREAKING: 0 commits (MAJOR bumps)
  docs:     4 commits (no bump)
  chore:    1 commit (no bump)
  security: 1 commit (MINOR bumps)
  Total:    20 commits

Feature Analysis:
  Completed:   7
  In Progress: 6
  Planned:     13
  Total:       26

Epic Analysis:
  Completed:   1
  In Progress: 2
  Total:       8

Version Calculation:
  Bump Type: MINOR (new features)
  Next Version: v0.2.0

=== Summary ===
Current:  v0.1.0
Next:     v0.2.0
Commits:  20
Features: 7/26
Epics:    1/8

Update PLAN.md with new version counters? (y/n)
```

**When to run:**
- After completing an epic
- Before creating a release
- When updating PLAN.md
- During sprint planning

**Integration with CI/CD:**
```yaml
# .github/workflows/version-check.yml
name: Version Check
on: [pull_request]
jobs:
  version:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check version
        run: ./scripts/update_version.sh
```

## Pre-publish Cleanup

### `clean_for_github.sh`

Removes local build/runtime artifacts before publishing to GitHub.

**Usage:**  
```bash
# Dry-run (default)
./scripts/clean_for_github.sh --dry-run

# Apply cleanup
./scripts/clean_for_github.sh --apply

# Apply cleanup + local-only artifacts
./scripts/clean_for_github.sh --apply --include-local
```

**Cleanup scope (default):**
- `.build/`, `.swiftpm/`, `logs/`, docs build directories
- generated artifacts (`*.log`, `*.tmp`, `*.profdata`, `default.metallib`)
- local runtime DBs (`*.db`, `*.sqlite`, `*.sqlite3`, `auth.sqlite3`)

**Optional scope with `--include-local`:**
- `.archive/`, `.claude/`, `.gemini.md`
- root helper files like `test_*.swift`

The script skips files/directories tracked by git to avoid deleting committed project content.

## Future Scripts

### `run_tests.sh` (Planned)
Run full test suite with coverage reporting.

### `benchmark.sh` (Planned)
Run performance benchmarks on all verified models.

### `security_audit.sh` (Planned)
Run dependency and CWE security audits.

### `release.sh` (Planned)
Automated release process: version bump, changelog, git tag, GitHub release.
