#!/usr/bin/env bash
# Save your work off this PC: commit everything and push it to GitHub.
# Run it after every finished step of BUILD_PLAN.md, and at least every 30 minutes
# while you code. The power can go at any second, and only what is on GitHub is
# safe from a damaged disk.
#
#   bash scripts/checkpoint.sh "Day 3: page index built"
cd "$(dirname "$0")/.." || exit 1
message="${1:-checkpoint}"

git add -A
if ! git diff --cached --quiet; then
    git commit -q -m "$message" || {
        echo "Commit failed -- read the message above (Appendix C of BUILD_PLAN.md)."
        exit 1
    }
fi

if ! git push -q; then
    echo "Committed on this PC, but the push failed (no internet?). Run this again later."
    exit 1
fi
echo "Safe on GitHub: $(git log -1 --format='%h %s')"
