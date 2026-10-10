#!/usr/bin/env bash
set -euo pipefail

archive_root="${STAGE3_ARCHIVE_DIR:-$HOME/stage3-archives}"
mkdir -p "$archive_root"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$archive_root/stage3-results-$stamp.zip"

python -m zipfile -c "$archive" runs reports environment.txt
sha256sum "$archive" | tee "$archive.sha256"
printf 'Archive excludes model weights and caches: %s\n' "$archive"
