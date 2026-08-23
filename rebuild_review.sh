#!/bin/zsh
# Rebuild the review branches so a diff-based reviewer sees the project at a reviewable SIZE.
#
# WHY THIS EXISTS -- two failures, both real, both silent:
#   1. 'review' was one squashed commit, then 28 commits accumulated on it. A reviewer scoped
#      itself to the recent ones and reported the target as a "documentation-only diff".
#   2. Squashing fixed that and created the opposite problem: 'main' is a single initial commit,
#      so the whole project reads as one 486-file / 139k-line addition and the reviewer refuses
#      it (limits: 500 files, 8,000 lines). configs/tokenizer.json alone is 19,728 lines.
#
# THE FIX: build each review target as a TWO-COMMIT chain off main --
#      main -> <base: full tree minus the target files> -> <head: full tree>
# so diff(base...head) is exactly the target, while the base still carries the whole project as
# context the reviewer can open. Three-dot diff uses the merge base, which is why the base must be
# an ANCESTOR of the head and not a sibling branch off main (a sibling gives back the 139k diff).
#
# Run this immediately before any review, then:
#      /ultrareview review-code-base     # while on 'review'       -> the code, ~6.7k lines
#      /ultrareview review-docs-base     # while on 'review-docs'  -> report.md, ~4k lines
set -e
cd "$(dirname "$0")"
git branch -f submission review 2>/dev/null || true      # keep whatever history 'review' had
git checkout -q --detach                                 # a checked-out branch cannot be force-updated
TMPIDX="$(mktemp -t tlabidx)"; trap 'rm -f "$TMPIDX"' EXIT
FULLTREE=$(git write-tree)                               # current index == working tree

mk() {  # mk <base-branch> <head-branch> <files-to-exclude...>
  local base=$1 head=$2; shift 2
  rm -f "$TMPIDX"
  GIT_INDEX_FILE="$TMPIDX" git read-tree "$FULLTREE"
  GIT_INDEX_FILE="$TMPIDX" git rm --cached -q --ignore-unmatch "$@"
  local tree c1 c2
  tree=$(GIT_INDEX_FILE="$TMPIDX" git write-tree)
  c1=$(git commit-tree "$tree" -p main -m "review base for '$head': whole project minus the files under review")
  c2=$(git commit-tree "$FULLTREE" -p "$c1" -m "T-Lab looped-transformer: the files under review on '$head' (rebuilt $(date '+%Y-%m-%d %H:%M'))")
  git branch -f "$base" "$c1"
  git branch -f "$head" "$c2"
  printf "  %-16s -> %-12s %s\n" "$base" "$head" "$(git diff --shortstat $base...$head)"
}

mk review-code-base review      $(git ls-files 'src/*.py') kaggle/main.py
mk review-docs-base review-docs report.md

git checkout -q review
echo "on branch: $(git rev-parse --abbrev-ref HEAD)"
