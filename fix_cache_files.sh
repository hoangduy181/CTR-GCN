#!/bin/bash
# Script to remove Python cache files from git tracking

echo "Removing __pycache__ directories from git tracking..."
find . -type d -name "__pycache__" | while read dir; do
    git rm -r --cached "$dir" 2>/dev/null || true
done

echo "Removing .pyc files from git tracking..."
find . -name "*.pyc" | while read file; do
    git rm --cached "$file" 2>/dev/null || true
done

echo "Unstaging any staged cache files..."
git reset HEAD -- "**/__pycache__/**" "**/*.pyc" 2>/dev/null || true

echo "Done! Check git status to verify."
