#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

BIN=${BIN:-./shmem_mega_moe}
RANK_SIZE=${RANK_SIZE:-2}
IP_PORT=${IP_PORT:-tcp://127.0.0.1:8766}
NPU_NUM=${NPU_NUM:-8}
FIRST_NPU=${FIRST_NPU:-0}
MODE=${MODE:-arch35_e4m3}
TOKEN_COUNT=${TOKEN_COUNT:-4}
MODEL_DIM=${MODEL_DIM:-4096}
FFN_DIM=${FFN_DIM:-1024}
EXPERTS_PER_TOKEN=${EXPERTS_PER_TOKEN:-2}
LOCAL_EXPERT_COUNT=${LOCAL_EXPERT_COUNT:-1}
MAX_RECEIVED_TOKENS=${MAX_RECEIVED_TOKENS:-0}
WARMUP=${WARMUP:-0}
LOOP=${LOOP:-1}
AIC_NUM=${AIC_NUM:-}
AIV_NUM=${AIV_NUM:-}

if [[ -n "${AIC_NUM}" ]]; then
  export SHMEM_MEGA_MOE_AIC_NUM=${AIC_NUM}
fi
if [[ -n "${AIV_NUM}" ]]; then
  export SHMEM_MEGA_MOE_AIV_NUM=${AIV_NUM}
fi

pids=()
for ((rank = 0; rank < RANK_SIZE; rank++)); do
  "${BIN}" "${RANK_SIZE}" "${rank}" "${IP_PORT}" "${NPU_NUM}" "${FIRST_NPU}" \
    "${MODE}" "${TOKEN_COUNT}" "${MODEL_DIM}" "${FFN_DIM}" "${EXPERTS_PER_TOKEN}" \
    "${LOCAL_EXPERT_COUNT}" "${MAX_RECEIVED_TOKENS}" "${WARMUP}" "${LOOP}" &
  pids+=("$!")
done

status=0
active_pids=("${pids[@]}")
while ((${#active_pids[@]} > 0)); do
  running_pids=" $(jobs -pr | tr '\n' ' ')"
  next_pids=()
  for pid in "${active_pids[@]}"; do
    if [[ ${running_pids} == *" ${pid} "* ]]; then
      next_pids+=("${pid}")
    elif wait "${pid}"; then
      continue
    else
      status=$?
      break 2
    fi
  done
  if ((${#next_pids[@]} == 0)); then
    break
  fi
  active_pids=("${next_pids[@]}")
  sleep 0.1
done

if ((status != 0)); then
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
fi

exit "${status}"
