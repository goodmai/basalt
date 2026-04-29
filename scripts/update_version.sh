#!/bin/bash
# Gem Version Counter & Updater
# Automatically calculates next version based on commit history and epic completion

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Gem Version Counter ===${NC}\n"

# Get current version from PLAN.md
CURRENT_VERSION=$(grep "Current Status:" PLAN.md | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' || echo "v0.1.0")
echo -e "Current Version: ${GREEN}${CURRENT_VERSION}${NC}"

# Parse version components
MAJOR=$(echo $CURRENT_VERSION | cut -d'.' -f1 | sed 's/v//')
MINOR=$(echo $CURRENT_VERSION | cut -d'.' -f2)
PATCH=$(echo $CURRENT_VERSION | cut -d'.' -f3)

echo -e "  MAJOR: $MAJOR"
echo -e "  MINOR: $MINOR"
echo -e "  PATCH: $PATCH\n"

# Count commits by type
echo -e "${BLUE}Commit Analysis:${NC}"
FEAT_COMMITS=$(git log --oneline --all | grep -c "^[a-f0-9]* feat:" || echo "0")
FIX_COMMITS=$(git log --oneline --all | grep -c "^[a-f0-9]* fix:" || echo "0")
BREAKING_COMMITS=$(git log --oneline --all | grep -c "BREAKING:" || echo "0")
DOCS_COMMITS=$(git log --oneline --all | grep -c "^[a-f0-9]* docs:" || echo "0")
CHORE_COMMITS=$(git log --oneline --all | grep -c "^[a-f0-9]* chore:" || echo "0")
SECURITY_COMMITS=$(git log --online --all | grep -c "^[a-f0-9]* security:" || echo "0")
TOTAL_COMMITS=$(git log --oneline --all | wc -l | tr -d ' ')

echo -e "  feat:     ${GREEN}${FEAT_COMMITS}${NC} commits (MINOR bumps)"
echo -e "  fix:      ${YELLOW}${FIX_COMMITS}${NC} commits (PATCH bumps)"
echo -e "  BREAKING: ${RED}${BREAKING_COMMITS}${NC} commits (MAJOR bumps)"
echo -e "  docs:     ${DOCS_COMMITS} commits (no bump)"
echo -e "  chore:    ${CHORE_COMMITS} commits (no bump)"
echo -e "  security: ${SECURITY_COMMITS} commits (MINOR bumps)"
echo -e "  ${BLUE}Total:    ${TOTAL_COMMITS} commits${NC}\n"

# Count features
echo -e "${BLUE}Feature Analysis:${NC}"
COMPLETED_FEATURES=$(grep -c "\[x\]" PLAN.md || echo "0")
IN_PROGRESS_FEATURES=$(grep -c "🔄" PLAN.md || echo "0")
PLANNED_FEATURES=$(grep -c "📋" PLAN.md || echo "0")
TOTAL_FEATURES=$((COMPLETED_FEATURES + IN_PROGRESS_FEATURES + PLANNED_FEATURES))

echo -e "  Completed:   ${GREEN}${COMPLETED_FEATURES}${NC}"
echo -e "  In Progress: ${YELLOW}${IN_PROGRESS_FEATURES}${NC}"
echo -e "  Planned:     ${PLANNED_FEATURES}"
echo -e "  ${BLUE}Total:       ${TOTAL_FEATURES}${NC}\n"

# Count completed epics
echo -e "${BLUE}Epic Analysis:${NC}"
COMPLETED_EPICS=$(grep -c "## Epic.*✅.*COMPLETED" PLAN.md || echo "0")
IN_PROGRESS_EPICS=$(grep -c "## Epic.*🔄.*IN PROGRESS" PLAN.md || echo "0")
TOTAL_EPICS=$(grep -c "## Epic [0-9]:" PLAN.md || echo "0")

echo -e "  Completed:   ${GREEN}${COMPLETED_EPICS}${NC}"
echo -e "  In Progress: ${YELLOW}${IN_PROGRESS_EPICS}${NC}"
echo -e "  ${BLUE}Total:       ${TOTAL_EPICS}${NC}\n"

# Calculate next version
echo -e "${BLUE}Version Calculation:${NC}"

# Check for breaking changes
if [ $BREAKING_COMMITS -gt 0 ]; then
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    BUMP_TYPE="MAJOR (breaking changes)"
# Check for new features
elif [ $FEAT_COMMITS -gt 0 ] || [ $SECURITY_COMMITS -gt 0 ]; then
    MINOR=$((MINOR + 1))
    PATCH=0
    BUMP_TYPE="MINOR (new features)"
# Check for bug fixes
elif [ $FIX_COMMITS -gt 0 ]; then
    PATCH=$((PATCH + 1))
    BUMP_TYPE="PATCH (bug fixes)"
else
    BUMP_TYPE="NONE (no version bump needed)"
fi

NEXT_VERSION="v${MAJOR}.${MINOR}.${PATCH}"

echo -e "  Bump Type: ${YELLOW}${BUMP_TYPE}${NC}"
echo -e "  Next Version: ${GREEN}${NEXT_VERSION}${NC}\n"

# Suggest version based on epic completion
if [ $COMPLETED_EPICS -gt 1 ]; then
    SUGGESTED_MINOR=$((COMPLETED_EPICS - 1))
    SUGGESTED_VERSION="v0.${SUGGESTED_MINOR}.0"
    echo -e "${YELLOW}Suggestion: Based on ${COMPLETED_EPICS} completed epics, consider ${SUGGESTED_VERSION}${NC}\n"
fi

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "Current:  ${CURRENT_VERSION}"
echo -e "Next:     ${GREEN}${NEXT_VERSION}${NC}"
echo -e "Commits:  ${TOTAL_COMMITS}"
echo -e "Features: ${COMPLETED_FEATURES}/${TOTAL_FEATURES}"
echo -e "Epics:    ${COMPLETED_EPICS}/${TOTAL_EPICS}\n"

# Ask if user wants to update PLAN.md
read -p "Update PLAN.md with new version counters? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Updating PLAN.md...${NC}"

    # Update Current Status line
    sed -i.bak "s/Current Status: v[0-9]*\.[0-9]*\.[0-9]*/Current Status: ${NEXT_VERSION}/" PLAN.md

    # Update Commit Counter
    sed -i.bak "s/\*\*Total Commits:\*\* [0-9]*/\*\*Total Commits:\*\* ${TOTAL_COMMITS}/" PLAN.md

    # Update Feature Counter
    sed -i.bak "s/\*\*Completed Features:\*\* [0-9]*/\*\*Completed Features:\*\* ${COMPLETED_FEATURES}/" PLAN.md

    rm PLAN.md.bak

    echo -e "${GREEN}✓ PLAN.md updated successfully${NC}"
    echo -e "${YELLOW}Don't forget to commit the changes!${NC}"
else
    echo -e "${YELLOW}Skipped PLAN.md update${NC}"
fi
