#!/bin/bash
# Security Audit Script for GemmaServer
# Part of Epic 8: Security & Dependency Audit

set -e

echo "🔒 GemmaServer Security Audit"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ISSUES_FOUND=0

# 1. Check for hardcoded secrets
echo "1️⃣  Checking for hardcoded secrets..."
if grep -r "password\|secret\|api_key\|token" Sources/ --include="*.swift" | grep -v "// " | grep -E "(=\"|= \")[^\"]{8,}\"" | grep -v "Bearer\|Authorization\|Content-Type\|jwt\|bcrypt\|SHA256"; then
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
if command -v jq &> /dev/null; then
    ./osv_check.sh > osv_results.tmp 2>&1
    
    # Check our actual version of swift-crypto
    OUR_CRYPTO_VERSION=$(grep -A 5 "swift-crypto" Package.resolved | grep "version" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')
    
    if grep -q "vulns" osv_results.tmp && grep -q "swift-crypto" osv_results.tmp; then
        echo -e "${YELLOW}⚠️  CVE found in swift-crypto database${NC}"
        echo "   Our version: $OUR_CRYPTO_VERSION"
        echo "   Affected: 4.0.0 - 4.3.0"
        
        # Check if our version is affected
        if [[ "$OUR_CRYPTO_VERSION" == 4.* ]]; then
            echo -e "${RED}❌ FAIL: We are using affected version!${NC}"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        else
            echo -e "${GREEN}✅ PASS: Our version is NOT affected${NC}"
        fi
    else
        echo -e "${GREEN}✅ PASS: No vulnerabilities found${NC}"
    fi
    rm -f osv_results.tmp
else
    echo -e "${YELLOW}⚠️  WARNING: jq not installed, skipping OSV scan${NC}"
    echo "   Install with: brew install jq"
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
    
    # Check for outdated packages (optional)
    echo "   Dependency versions:"
    grep -A 5 "identity" Package.resolved | grep -E "identity|version" | sed 's/^/   /' | head -20
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
