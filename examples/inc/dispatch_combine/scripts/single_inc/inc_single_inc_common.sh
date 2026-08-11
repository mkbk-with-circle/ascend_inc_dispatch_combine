#!/usr/bin/env bash
# Shared placement, topology and accelerator-idle checks for single-INC gates.

inc_single_worker_phys() {
  local workers=$1
  local inc_phy=${INC_SINGLE_INC_PHY:-0}
  [[ "$workers" =~ ^[1-9][0-9]*$ && "$inc_phy" =~ ^[0-9]+$ ]] || {
    echo "invalid workers/INC Phy-ID: workers=$workers inc_phy=$inc_phy" >&2
    return 2
  }
  # A hardware profile may qualify a different peer set for every scale.  Use
  # that scale-specific map first, then retain the generic one-shot override
  # for ad-hoc qualification runs.
  local profile_worker_var="INC_SINGLE_INC_WORKER_PHYS_W${workers}"
  local profile_worker_phys=${!profile_worker_var:-}
  local selected_worker_phys=${profile_worker_phys:-${INC_SINGLE_INC_WORKER_PHYS:-}}
  if [[ -n "$selected_worker_phys" ]]; then
    local requested=()
    read -r -a requested <<<"$selected_worker_phys"
    [[ "${#requested[@]}" -eq "$workers" ]] || {
      echo "INC_SINGLE_INC_WORKER_PHYS count does not match workers=$workers" >&2
      return 2
    }
    local seen=" " phy
    for phy in "${requested[@]}"; do
      [[ "$phy" =~ ^[0-9]+$ && "$phy" -ne "$inc_phy" ]] || {
        echo "invalid single-INC worker Phy-ID: $phy" >&2; return 2;
      }
      [[ "$seen" != *" $phy "* ]] || {
        echo "duplicate single-INC worker Phy-ID: $phy" >&2; return 2;
      }
      seen+="$phy "
    done
    echo "${requested[*]}"
    return 0
  fi
  # Portable default: discover peers in the live HCCS fabric.  A measured
  # machine profile can pin a preferred set through the scale-specific
  # WORKER_PHYS variables above without rebuilding anything.
  local topo row candidates=() col relation
  topo=$(npu-smi info -t topo -i 0 2>/dev/null) || return 2
  row=$(awk -v phy_key="Phy-ID${inc_phy}" -v npu_key="NPU${inc_phy}" \
    '$1 == phy_key || $1 == npu_key {for (i=2;i<=NF;i++) if ($i=="X") {print; exit}}' \
    <<<"$topo")
  [[ -n "$row" ]] || { echo "cannot discover topology row for Phy${inc_phy}" >&2; return 2; }
  local fields
  read -r -a fields <<<"$row"
  for ((col=1; col<${#fields[@]}; ++col)); do
    relation=${fields[$col]}
    [[ "$relation" == "HCCS_SW" || "$relation" == "HCCS" ]] && \
      candidates+=("$((col - 1))")
  done
  [[ "${#candidates[@]}" -ge "$workers" ]] || {
    echo "only ${#candidates[@]} HCCS_SW peers found for workers=$workers INC Phy${inc_phy}" >&2
    return 2
  }
  echo "${candidates[*]:0:workers}"
}

inc_single_pe_map() {
  local workers=$1
  local inc_phy=${INC_SINGLE_INC_PHY:-0}
  local phys
  read -r -a phys <<<"$(inc_single_worker_phys "$workers")"
  local entries=()
  local rank
  for ((rank=0; rank<workers; ++rank)); do
    entries+=("${rank}:${phys[$rank]}")
  done
  entries+=("${workers}:${inc_phy}")
  local IFS=,
  echo "${entries[*]}"
}

inc_single_verify_live_topology() {
  local workers=$1
  local inc_phy=${INC_SINGLE_INC_PHY:-0}
  local topo
  topo=$(npu-smi info -t topo -i 0)
  local row
  row=$(awk -v phy_key="Phy-ID${inc_phy}" -v npu_key="NPU${inc_phy}" \
    '$1 == phy_key || $1 == npu_key {for (i=2;i<=NF;i++) if ($i=="X") {print; exit}}' \
    <<<"$topo")
  [[ -n "$row" ]] || {
    echo "[REFUSED] cannot read the live Phy-ID${inc_phy} topology row" >&2
    return 20
  }

  local expected_var="INC_SINGLE_INC_EXPECTED_RELATIONS_W${workers}"
  local expected_text=${!expected_var:-}
  local expected=()
  [[ -n "$expected_text" ]] && read -r -a expected <<<"$expected_text"
  local worker_phys=()
  read -r -a worker_phys <<<"$(inc_single_worker_phys "$workers")"
  [[ "${#expected[@]}" -eq 0 || "${#expected[@]}" -eq "${#worker_phys[@]}" ]] || {
    echo "[REFUSED] expected-relation count does not match workers=$workers" >&2
    return 22
  }

  local index phy relation wanted
  for index in "${!worker_phys[@]}"; do
    phy=${worker_phys[$index]}
    relation=$(awk -v col=$((phy + 2)) '{print $col}' <<<"$row")
    wanted=${expected[$index]:-HCCS}
    if [[ "$wanted" == "HCCS" ]]; then
      [[ "$relation" == "HCCS" || "$relation" == "HCCS_SW" ]] && continue
    elif [[ "$relation" == "$wanted" ]]; then
      continue
    fi
    if [[ "$relation" != "$wanted" ]]; then
      echo "[REFUSED] INC Phy${inc_phy} -> worker Phy${phy} is ${relation:-unknown}, expected $wanted" >&2
      return 21
    fi
  done
  echo "SINGLE_INC_TOPOLOGY_OK workers=$workers inc_phy=$inc_phy worker_phys=${worker_phys[*]} relations=${expected_text:-HCCS-compatible}"
}

inc_single_npu_idle() {
  local info total_cards idle_cards
  info=$(npu-smi info 2>/dev/null) || return 1
  total_cards=$(npu-smi info -l 2>/dev/null | awk -F: '/Total Count/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
  [[ "$total_cards" =~ ^[0-9]+$ && "$total_cards" -gt 0 ]] || return 1
  idle_cards=$(grep -c 'No running processes found in NPU' <<<"$info" || true)
  [[ "$idle_cards" -eq "$total_cards" ]] || return 1
  local exe base
  for exe in /proc/[0-9]*/exe; do
    exe=$(readlink "$exe" 2>/dev/null || true)
    base=${exe##*/}
    case "$base" in
      inc_dc_*|dispatch_doubleplane*|combine_doubleplane*) return 1 ;;
    esac
  done
  return 0
}

inc_single_list_operator_processes() {
  local link exe base pid
  for link in /proc/[0-9]*/exe; do
    exe=$(readlink "$link" 2>/dev/null || true)
    base=${exe##*/}
    case "$base" in
      inc_dc_*|dispatch_doubleplane*|combine_doubleplane*)
        pid=${link#/proc/}; pid=${pid%/exe}
        printf '%s %s\n' "$pid" "$exe" >&2
        ;;
    esac
  done
}

inc_single_wait_for_npu_idle() {
  local label=${1:-single_inc_case}
  local poll_sec=${INC_NPU_IDLE_POLL_SEC:-5}
  local waited=0
  while ! inc_single_npu_idle; do
    if (( waited % 10 == 0 )); then
      echo "WAIT_NPU_IDLE label=$label waited_sec=$waited" >&2
      inc_single_list_operator_processes || true
    fi
    sleep "$poll_sec"
    waited=$((waited + poll_sec))
  done
  echo "NPU_IDLE_OK label=$label waited_sec=$waited" >&2
}

inc_single_source_cann() {
  local cann_home=${ASCEND_HOME_PATH:-}
  if [[ -z "$cann_home" || ! -f "$cann_home/set_env.sh" ]]; then
    if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
      cann_home=/usr/local/Ascend/cann-9.0.0
    elif [[ -f /usr/local/Ascend/cann-8.5.0/set_env.sh ]]; then
      cann_home=/usr/local/Ascend/cann-8.5.0
    else
      echo "[REFUSED] no supported CANN set_env.sh found" >&2
      return 23
    fi
  fi
  # shellcheck source=/dev/null
  source "$cann_home/set_env.sh"
  export ASCEND_HOME_PATH="$cann_home"
}
