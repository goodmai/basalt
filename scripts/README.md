# Basalt Scripts

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
=== Basalt Version Counter ===

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

### `cleanup.swift`

One cleanup script with three modes. Every mode is a dry run unless `--apply` is given,
and nothing tracked by git is ever removed.

**Usage:**  
```bash
# Pre-publish: list build/runtime artifacts (default mode is release)
./scripts/cleanup.swift --mode=release

# Remove them, plus local-only artifacts
./scripts/cleanup.swift --mode=release --apply --include-local

# Daily rotation: remove files older than 24 h from logs/, images/, reports/
./scripts/cleanup.swift --mode=daily --apply
./scripts/cleanup.swift --mode=daily --apply --max-age-hours=0   # everything untracked

# Both
./scripts/cleanup.swift --mode=all --apply
```

**Release scope:**
- `.build/`, `.swiftpm/`, `logs/`, docs build directories
- generated artifacts (`*.log`, `*.tmp`, `*.swp`, `*.profdata`, `.DS_Store`, `benchmark_*.json`)
- local runtime DBs in the root (`*.db`, `*.sqlite`, `*.sqlite3`, `auth.sqlite3`)
- with `--include-local`: `.archive/`, `.claude/`, `.gemini.md`, root `test_*.swift`

`default.metallib` is kept: the MLX runtime loads it from the repo root.

## Future Scripts

### `run_tests.sh` (Planned)
Run full test suite with coverage reporting.

### `benchmark.sh` (Planned)
Run performance benchmarks on all verified models.

### `release.sh` (Planned)
Automated release process: version bump, changelog, git tag, GitHub release.
