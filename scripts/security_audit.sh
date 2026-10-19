#!/bin/bash
# Security audit for Basalt: source greps, repository hygiene and an OSV scan of every
# pinned dependency in Package.resolved.
#
# Usage:
#   ./scripts/security_audit.sh                        full audit; exit 1 on any failed check
#   ./scripts/security_audit.sh --osv-only [RESOLVED]  only the OSV scan (default Package.resolved)

set -e

# Query OSV for every pin in $1 (Package.resolved).
#
# OSV's SwiftPM ecosystem is "SwiftURL", and it keys packages by their repository URL
# without the scheme — "github.com/apple/swift-nio", not "swift-nio". Asking for the bare
# name matches nothing: every query comes back {} and the scan passes by never checking
# anything. For the same reason a query that fails (network, HTTP error, unparsable reply)
# fails the scan instead of counting as "no vulnerabilities".
#
# Returns 1 when a dependency is affected or could not be checked, so CI stops rather than
# filing it in a report nobody reads.
osv_scan() {
    local resolved="${1:-Package.resolved}"
    if [[ ! -f "$resolved" ]]; then
        echo "No $resolved — nothing to scan" >&2
        return 1
    fi
    local tool
    for tool in curl jq python3; do
        if ! command -v "$tool" &> /dev/null; then
            echo "OSV scan needs $tool (brew install $tool)" >&2
            return 1
        fi
    done

    local found=0 unchecked=0 checked=0 name url version pkg response ids
    while IFS=$'\t' read -r name url version; do
        if [[ -z "$url" || -z "$version" ]]; then
            unchecked=1
            echo "UNCHECKED   $name (no location or version in $resolved)"
            continue
        fi
        pkg="${url#https://}"
        pkg="${pkg%.git}"

        if ! response=$(curl -sSf -m 30 "https://api.osv.dev/v1/query" \
                -d "{\"package\":{\"name\":\"$pkg\",\"ecosystem\":\"SwiftURL\"},\"version\":\"$version\"}") ||
           ! printf '%s' "$response" | jq -e 'type == "object"' > /dev/null 2>&1; then
            unchecked=1
            echo "UNCHECKED   $name @ $version (OSV query failed)"
            continue
        fi

        checked=$((checked + 1))
        ids=$(printf '%s' "$response" | jq -r '.vulns[]?.id')
        if [[ -n "$ids" ]]; then
            found=1
            echo "VULNERABLE  $name @ $version"
            printf '%s' "$response" | jq -r '.vulns[] | "    \(.id)  \(.summary // "no summary")"'
        else
            echo "ok          $name @ $version"
        fi
    done < <(python3 -c '
import json, sys
pins = json.load(open(sys.argv[1]))["pins"]
for p in pins:
    state = p.get("state", {})
    # Revision-only pins get an empty version: OSV keys versions, so they are reported UNCHECKED.
    print("\t".join([p.get("identity", "?"), p.get("location", ""), state.get("version") or ""]))
' "$resolved")

    echo
    # Zero pins means the parse failed (Package.resolved v1, broken JSON): nothing was checked.
    if (( checked == 0 )); then
        echo "OSV checked no dependency — could not read pins from $resolved."
        return 1
    fi
    if (( found )); then
        echo "OSV reported vulnerable dependencies — see above."
        return 1
    fi
    if (( unchecked )); then
        echo "OSV could not check every dependency — see above."
        return 1
    fi
    echo "OSV: no known vulnerabilities in the pinned dependencies."
}

if [[ "${1:-}" == "--osv-only" ]]; then
    osv_scan "${2:-Package.resolved}"
    exit $?
fi

echo "🔒 Basalt Security Audit"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
# Plain text when not on a terminal (CI logs, tee into the report).
[ -t 1 ] || { RED=''; GREEN=''; YELLOW=''; NC=''; }

ISSUES_FOUND=0

# 1. Check for hardcoded secrets
echo "1️⃣  Checking for hardcoded secrets..."
# Look for credential *values*, not identifiers containing "token": the old
# pattern matched every `case maxTokens = "max_tokens"` in a CodingKeys enum and
# so failed on every run, which is the fastest way to teach people to ignore a
# security check. Two things are worth flagging — a literal that looks like a
# real key, and a credential-ish name assigned a long literal (never a `case`,
# which is JSON field mapping).
CRED_VALUE='sk-ant-[A-Za-z0-9_-]{20}|sk-[A-Za-z0-9]{32}|gh[pousr]_[A-Za-z0-9]{30}|github_pat_[A-Za-z0-9_]{30}|hf_[A-Za-z0-9]{30}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{20}|-----BEGIN (RSA|OPENSSH|EC|PRIVATE) PRIVATE KEY'
CRED_ASSIGN='(let|var)[[:space:]]+[A-Za-z_]*([Pp]assword|[Ss]ecret|[Aa]pi[Kk]ey|[Aa]uth[Tt]oken)[A-Za-z_]*[[:space:]]*(:[^=]*)?=[[:space:]]*"[^"]{12,}"'

if grep -rInE "$CRED_VALUE" Sources/ --include="*.swift" ||
   grep -rInE "$CRED_ASSIGN" Sources/ --include="*.swift" | grep -v "^[^:]*:[0-9]*: *case "; then
    echo -e "${RED}❌ FAIL: Potential hardcoded secrets found${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
    echo -e "${GREEN}✅ PASS: No hardcoded secrets detected${NC}"
fi
echo ""

# 2. Check for SQL injection vulnerabilities
echo "2️⃣  Checking for SQL injection vulnerabilities..."
if grep -r "SELECT.*\\$\|INSERT.*\\$\|UPDATE.*\\$\|DELETE.*\\$" Sources/ --include="*.swift" | grep -v "//"; then
    echo -e "${RED}❌ FAIL: Potential SQL injection via string interpolation${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
    echo -e "${GREEN}✅ PASS: No SQL string interpolation found${NC}"
fi
echo ""

# 3. Check for unsafe force unwraps in production code
echo "3️⃣  Checking for force unwraps (!)..."
FORCE_UNWRAPS=$(grep -r "!" Sources/ --include="*.swift" | grep -v "// " | grep -v "!=" | grep -v "try!" | wc -l)
if [ "$FORCE_UNWRAPS" -gt 20 ]; then
    echo -e "${YELLOW}⚠️  WARNING: Found $FORCE_UNWRAPS potential force unwraps${NC}"
    echo "   (Review manually - some may be safe)"
else
    echo -e "${GREEN}✅ PASS: Minimal force unwraps ($FORCE_UNWRAPS)${NC}"
fi
echo ""

# 4. Dependency vulnerability scan (OSV)
echo "4️⃣  Running dependency vulnerability scan (OSV)..."
if osv_scan Package.resolved; then
    echo -e "${GREEN}✅ PASS: No known vulnerabilities in pinned dependencies${NC}"
else
    echo -e "${RED}❌ FAIL: OSV scan found affected or unchecked dependencies${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# 5. Check for print/dump statements (information leakage)
echo "5️⃣  Checking for debug print statements..."
DEBUG_PRINTS=$(grep -r "print(\|dump(\|debugPrint(" Sources/ --include="*.swift" | grep -v "// " | wc -l)
if [ "$DEBUG_PRINTS" -gt 5 ]; then
    echo -e "${YELLOW}⚠️  WARNING: Found $DEBUG_PRINTS print/dump statements${NC}"
    echo "   (Use logger instead for production code)"
else
    echo -e "${GREEN}✅ PASS: Minimal print statements ($DEBUG_PRINTS)${NC}"
fi
echo ""

# 6. Check for weak cryptography
echo "6️⃣  Checking for weak cryptography..."
if grep -r "MD5\|SHA1\|DES\|RC4" Sources/ --include="*.swift" | grep -v "// "; then
    echo -e "${RED}❌ FAIL: Weak cryptographic algorithms detected${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
    echo -e "${GREEN}✅ PASS: No weak crypto algorithms found${NC}"
fi
echo ""

# 7. Check for HTTP (non-HTTPS) URLs
echo "7️⃣  Checking for HTTP (insecure) URLs..."
if grep -r "http://" Sources/ --include="*.swift" | grep -v "// " | grep -v "localhost\|127.0.0.1"; then
    echo -e "${YELLOW}⚠️  WARNING: HTTP URLs found (should use HTTPS)${NC}"
else
    echo -e "${GREEN}✅ PASS: No insecure HTTP URLs found${NC}"
fi
echo ""

# 8. Check for environment variable exposure
echo "8️⃣  Checking .env files are in .gitignore..."
if grep -q "\.env" .gitignore; then
    echo -e "${GREEN}✅ PASS: .env files in .gitignore${NC}"
else
    echo -e "${RED}❌ FAIL: .env files not in .gitignore${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# 9. Check for sensitive files in git
echo "9️⃣  Checking for sensitive files in repository..."
SENSITIVE_FILES=$(git ls-files | grep -E "\.key$|\.pem$|\.p12$|credentials|secret" || true)
if [ -n "$SENSITIVE_FILES" ]; then
    echo -e "${RED}❌ FAIL: Sensitive files found in repository:${NC}"
    echo "$SENSITIVE_FILES"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
    echo -e "${GREEN}✅ PASS: No sensitive files in repository${NC}"
fi
echo ""

# 10. Swift Package dependency check
echo "🔟 Checking Swift Package dependencies..."
if [ -f "Package.resolved" ]; then
    DEPS_COUNT=$(grep -c "identity" Package.resolved || echo "0")
    echo -e "${GREEN}✅ Found $DEPS_COUNT dependencies in Package.resolved${NC}"
    # Per-dependency versions are listed by the OSV scan above.
else
    echo -e "${YELLOW}⚠️  WARNING: Package.resolved not found${NC}"
fi
echo ""

# Summary
echo "=============================="
echo "📊 Audit Summary"
echo "=============================="
if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: All security checks passed!${NC}"
    echo ""
    echo "Security score: 10/10 🏆"
    exit 0
else
    echo -e "${RED}❌ FAIL: $ISSUES_FOUND security issues found${NC}"
    echo ""
    echo "Please fix the issues above before deploying to production."
    exit 1
fi
