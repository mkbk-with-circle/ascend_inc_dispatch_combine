#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
# Smoke test: install built wheel and verify basic functionality.
#
# Usage: bash tests/package_smoke/run_install_smoke.sh [wheel_path]
#   If wheel_path is omitted, auto-discovers dist/cann_shmem-*.whl in the project root.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." &>/dev/null && pwd)"

# ---------- locate wheel ----------
if [ $# -ge 1 ]; then
    WHEEL="$1"
else
    # auto-discover
    wheels=("$PROJECT_ROOT"/dist/cann_shmem-*.whl)
    if [ ! -f "${wheels[0]}" ]; then
        echo "[ERROR] No wheel found under $PROJECT_ROOT/dist/ (cann_shmem-*.whl)"
        exit 1
    fi
    WHEEL="${wheels[0]}"
fi

echo "===== Smoke test wheel: $WHEEL ====="

# ---------- Step 1: install wheel ----------
echo "===== [1/4] Installing wheel ====="
pip install --force-reinstall "$WHEEL"
echo "===== Installation done ====="

# ---------- Step 2: import shmem ----------
echo "===== [2/4] Importing shmem module ====="
python3 -c "
import shmem
assert shmem._NATIVE_LOADED, 'Native bindings failed to load — import succeeded but SHMEM APIs are unavailable'
print('shmem imported successfully (native bindings loaded)')
"
echo -n "  version: "
shmem-config --version
echo "===== Import OK ====="

# ---------- Step 3: shmem-config --backend ----------
echo "===== [3/4] shmem-config --backend ====="
backend=$(shmem-config --backend 2>&1)
echo "Backend: $backend"
echo "===== --backend OK ====="

# ---------- Step 4: shmem-config --diagnose ----------
echo "===== [4/4] shmem-config --diagnose ====="
diagnose_output=$(shmem-config --diagnose 2>&1)
echo "$diagnose_output"

# verify it's valid JSON
echo "$diagnose_output" | python3 -c "import sys,json; json.load(sys.stdin)" || {
    echo "[ERROR] shmem-config --diagnose did not produce valid JSON"
    exit 1
}
echo "===== --diagnose OK ====="

# ---------- verify key CLI entry-points ----------
echo "===== Bonus: verify --ldflags, --rpath, --include, --lib, --version ====="
for cmd in --ldflags --rpath --include --lib --version; do
    echo -n "  $cmd -> "
    shmem-config "$cmd"
done
echo "===== All CLI checks passed ====="

echo "===== SMOKE TEST PASSED ====="
