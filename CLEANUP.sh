#!/bin/bash
# Project cleanup script
# Run this script to remove dead code and test artifacts
# Usage: bash CLEANUP.sh

echo "MLX Gem Project Cleanup Script"
echo "=============================="
echo ""
echo "This script will remove dead code and development artifacts."
echo "Press Ctrl+C to cancel."
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  exit 1
fi

# Delete dead Swift source
echo "Removing dead Swift source files..."
git rm Sources/Gem/Core/RenderCoordinator.swift

# Delete test scratch files
echo "Removing test scratch files..."
git rm test_mlx.swift test_json.swift test_urlsession.swift
git rm test_output.log test_result.log test_session.txt

# Delete one-off utility scripts
echo "Removing one-off utility scripts..."
git rm rename.py rename_script.py gen_checklists.py

# Delete old security documentation
echo "Removing superseded documentation..."
git rm SECURITY_OLD.md

# Delete Node.js files
echo "Removing Node.js files..."
git rm package.json package-lock.json

# Delete AI context file
echo "Removing AI assistant context..."
git rm .gemini.md

# Delete duplicate run scripts
echo "Removing duplicate run scripts..."
git rm run_26b_test.sh run_31b_test.sh run_26b_test_base.sh
git rm run_26b_base_bench.sh run_4b_bench.sh run_manual_test.sh
git rm run_manual_test2.sh run_rest_ui_test.sh run_test_and_capture.sh

# Stage .gitignore updates
echo "Staging .gitignore updates..."
git add .gitignore

# Commit
echo ""
echo "Creating commit..."
git commit -m "chore: remove dead code and clean up project root

- Remove RenderCoordinator.swift (unused Metal UI)
- Remove test scratch files and logs
- Remove one-off utility/refactoring scripts
- Remove SECURITY_OLD.md (superseded)
- Remove Node.js package files
- Remove .gemini.md (AI context)
- Remove duplicate root-level run scripts
- Update .gitignore with additional entries
- Keep all core source, documentation, and osv_check.sh"

echo ""
echo "Cleanup complete!"
git log --oneline -1
