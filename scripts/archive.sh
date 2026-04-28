#!/bin/bash
# Archive Manager for GemmaServer
# Archives completed tasks, plans, and reviews to .archive/ directory

set -e

ARCHIVE_DIR=".archive"
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== GemmaServer Archive Manager ===${NC}\n"

# Create archive directory structure
mkdir -p "${ARCHIVE_DIR}/plans"
mkdir -p "${ARCHIVE_DIR}/tasks"
mkdir -p "${ARCHIVE_DIR}/reviews"
mkdir -p "${ARCHIVE_DIR}/benchmarks"
mkdir -p "${ARCHIVE_DIR}/logs"

# Function to archive completed tasks from PLAN.md
archive_completed_tasks() {
    echo -e "${BLUE}Archiving completed tasks...${NC}"

    # Extract completed tasks (lines with [x])
    grep "\[x\]" PLAN.md > "${ARCHIVE_DIR}/tasks/completed_${TIMESTAMP}.md" || true

    local count=$(wc -l < "${ARCHIVE_DIR}/tasks/completed_${TIMESTAMP}.md" | tr -d ' ')
    echo -e "${GREEN}✓ Archived ${count} completed tasks${NC}"
}

# Function to archive old plan versions
archive_plan_version() {
    echo -e "${BLUE}Archiving PLAN.md version...${NC}"

    # Copy current PLAN.md with timestamp
    cp PLAN.md "${ARCHIVE_DIR}/plans/PLAN_${TIMESTAMP}.md"

    # Add git commit hash if in git repo
    if git rev-parse --git-dir > /dev/null 2>&1; then
        local commit=$(git rev-parse --short HEAD)
        echo "# Archived from commit: ${commit}" >> "${ARCHIVE_DIR}/plans/PLAN_${TIMESTAMP}.md"
    fi

    echo -e "${GREEN}✓ Archived PLAN.md${NC}"
}

# Function to archive code reviews
archive_reviews() {
    echo -e "${BLUE}Archiving code reviews...${NC}"

    # Find review comments in git log
    git log --grep="review:" --pretty=format:"%h - %s (%cd)" --date=short > \
        "${ARCHIVE_DIR}/reviews/reviews_${DATE}.txt" 2>/dev/null || true

    echo -e "${GREEN}✓ Archived code reviews${NC}"
}

# Function to archive benchmark results
archive_benchmarks() {
    echo -e "${BLUE}Archiving benchmark results...${NC}"

    # Move old benchmark JSON files
    find . -maxdepth 1 -name "benchmark_*.json" -o -name "res_*.json" -o -name "context_latency.json" | while read file; do
        if [ -f "$file" ]; then
            mv "$file" "${ARCHIVE_DIR}/benchmarks/"
            echo -e "  Moved: $(basename $file)"
        fi
    done

    echo -e "${GREEN}✓ Archived benchmark results${NC}"
}

# Function to archive logs
archive_logs() {
    echo -e "${BLUE}Archiving logs...${NC}"

    # Move old log files
    find . -maxdepth 1 -name "*.log" | while read file; do
        if [ -f "$file" ]; then
            mv "$file" "${ARCHIVE_DIR}/logs/"
            echo -e "  Moved: $(basename $file)"
        fi
    done

    echo -e "${GREEN}✓ Archived logs${NC}"
}

# Function to generate archive index
generate_index() {
    echo -e "${BLUE}Generating archive index...${NC}"

    cat > "${ARCHIVE_DIR}/INDEX.md" << INDEXEOF
# GemmaServer Archive Index

Generated: $(date)

## Plans
$(ls -1 ${ARCHIVE_DIR}/plans/ | sed 's/^/- /')

## Tasks
$(ls -1 ${ARCHIVE_DIR}/tasks/ | sed 's/^/- /')

## Reviews
$(ls -1 ${ARCHIVE_DIR}/reviews/ | sed 's/^/- /')

## Benchmarks
$(ls -1 ${ARCHIVE_DIR}/benchmarks/ | sed 's/^/- /')

## Logs
$(ls -1 ${ARCHIVE_DIR}/logs/ | sed 's/^/- /')

## Statistics
- Total Plans: $(ls -1 ${ARCHIVE_DIR}/plans/ | wc -l | tr -d ' ')
- Total Tasks: $(ls -1 ${ARCHIVE_DIR}/tasks/ | wc -l | tr -d ' ')
- Total Reviews: $(ls -1 ${ARCHIVE_DIR}/reviews/ | wc -l | tr -d ' ')
- Total Benchmarks: $(ls -1 ${ARCHIVE_DIR}/benchmarks/ | wc -l | tr -d ' ')
- Total Logs: $(ls -1 ${ARCHIVE_DIR}/logs/ | wc -l | tr -d ' ')
INDEXEOF

    echo -e "${GREEN}✓ Generated archive index${NC}"
}

# Function to clean old archives (keep last 30 days)
clean_old_archives() {
    echo -e "${BLUE}Cleaning old archives (>30 days)...${NC}"

    find "${ARCHIVE_DIR}" -type f -mtime +30 -delete

    echo -e "${GREEN}✓ Cleaned old archives${NC}"
}

# Main execution
case "${1:-all}" in
    tasks)
        archive_completed_tasks
        ;;
    plan)
        archive_plan_version
        ;;
    reviews)
        archive_reviews
        ;;
    benchmarks)
        archive_benchmarks
        ;;
    logs)
        archive_logs
        ;;
    clean)
        clean_old_archives
        ;;
    all)
        archive_completed_tasks
        archive_plan_version
        archive_reviews
        archive_benchmarks
        archive_logs
        generate_index
        ;;
    *)
        echo "Usage: $0 {tasks|plan|reviews|benchmarks|logs|clean|all}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=== Archive Complete ===${NC}"
echo -e "Archive location: ${ARCHIVE_DIR}/"
echo -e "View index: cat ${ARCHIVE_DIR}/INDEX.md"
