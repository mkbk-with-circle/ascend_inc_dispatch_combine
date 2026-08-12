#!/usr/bin/env bash
# Prepare the exact CATLASS source required by the qualified fusion kernel.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CATLASS_DIR=${INC_CATLASS_DIR:-$ROOT/3rdparty/catlass}
CATLASS_REPOSITORY=${INC_CATLASS_REPOSITORY:-https://gitcode.com/cann/catlass.git}
CATLASS_COMMIT=7d4c8401ae2b2aeb8a5786671e4fe7f53ca96c18
PATCH=$ROOT/examples/inc/fusion_kernel/third_party_patches/catlass_grouped_matmul_zero_m.patch
PATCHED_FILE=include/catlass/gemm/kernel/grouped_matmul_slice_m.hpp
PATCHED_FILE_SHA256=8d6c8b11a0826abac95974364012ff84b90042961c840c58a30dd75558d7cebb

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "neither sha256sum nor shasum is available" >&2
    return 127
  fi
}

[[ -f "$PATCH" ]] || { echo "missing CATLASS patch: $PATCH" >&2; exit 2; }

if [[ ! -d "$CATLASS_DIR/.git" ]]; then
  if [[ -e "$CATLASS_DIR" ]]; then
    echo "refusing non-Git CATLASS path: $CATLASS_DIR" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$CATLASS_DIR")"
  git clone "$CATLASS_REPOSITORY" "$CATLASS_DIR"
fi

head=$(git -C "$CATLASS_DIR" rev-parse HEAD)
if [[ "$head" != "$CATLASS_COMMIT" ]]; then
  [[ -z "$(git -C "$CATLASS_DIR" status --short --untracked-files=no)" ]] || {
    echo "refusing to switch a dirty CATLASS tree" >&2
    exit 2
  }
  git -C "$CATLASS_DIR" fetch origin "$CATLASS_COMMIT"
  git -C "$CATLASS_DIR" checkout --detach "$CATLASS_COMMIT"
fi

actual_file_sha256=$(sha256_file "$CATLASS_DIR/$PATCHED_FILE")
if [[ "$actual_file_sha256" == "$PATCHED_FILE_SHA256" ]]; then
  : # The exact pinned patch is already present.
elif [[ -n "$(git -C "$CATLASS_DIR" status --short --untracked-files=no)" ]]; then
  echo "refusing CATLASS tree with unrelated tracked changes: $CATLASS_DIR" >&2
  git -C "$CATLASS_DIR" status --short --untracked-files=no >&2
  exit 2
elif git -C "$CATLASS_DIR" apply --unidiff-zero --check "$PATCH" >/dev/null 2>&1; then
  git -C "$CATLASS_DIR" apply --unidiff-zero "$PATCH"
else
  echo "CATLASS patch is neither applicable nor already applied" >&2
  exit 2
fi

tracked_changes=$(
  git -C "$CATLASS_DIR" status --short --untracked-files=no |
    sed -n '/^[ MARC][MDARC] /p'
)
if [[ "$tracked_changes" != " M $PATCHED_FILE" ]]; then
  echo "unexpected CATLASS tracked state" >&2
  printf '%s\n' "$tracked_changes" >&2
  exit 2
fi

git -C "$CATLASS_DIR" diff --check
actual_file_sha256=$(sha256_file "$CATLASS_DIR/$PATCHED_FILE")
if [[ "$actual_file_sha256" != "$PATCHED_FILE_SHA256" ]]; then
  echo "CATLASS patched file does not match the pinned content" >&2
  exit 2
fi

echo "CATLASS_READY commit=$CATLASS_COMMIT patch=$(sha256_file "$PATCH") dir=$CATLASS_DIR"
