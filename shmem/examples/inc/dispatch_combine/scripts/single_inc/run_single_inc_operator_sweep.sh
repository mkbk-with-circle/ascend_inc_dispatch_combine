#!/usr/bin/env bash
# Full single-INC dispatch/combine volume matrix.  Large logical volumes are
# streamed through a bounded 256-MiB working set so the gate measures protocol
# scalability rather than allocator capacity.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
LOG_ROOT=${1:-/tmp/single-inc-operator-sweep-$(date +%Y%m%d-%H%M%S)}
REPORT=${2:-$LOG_ROOT/report.json}
mkdir -p "$LOG_ROOT"

read -r -a TOPKS <<< "${INC_SWEEP_TOPKS:-1 2 4 6 8}"
read -r -a WORKERS <<< "${INC_SWEEP_WORKERS:-2 4 8}"
# Probe the first performance-gated size before spending time on the full
# shape curve.  Report generation sorts by path, so execution order does not
# alter the matrix schema.
# Adjacent points differ by only 2x.  Keep 8 GiB first as a fail-early probe,
# then collect the monotonically increasing curve used by the report.
read -r -a VOLUMES <<< "${INC_SWEEP_VOLUMES:-8589934592 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912 1073741824 2147483648 4294967296}"
LEGAL_TOPK_ONLY=${INC_SWEEP_LEGAL_TOPK_ONLY:-0}
[[ "$LEGAL_TOPK_ONLY" == 0 || "$LEGAL_TOPK_ONLY" == 1 ]] || {
  echo "INC_SWEEP_LEGAL_TOPK_ONLY must be 0 or 1" >&2
  exit 2
}
WARMUP=${INC_SWEEP_WARMUP:-2}
REPEATS=${INC_SWEEP_REPEATS:-5}
[[ "$WARMUP" =~ ^[0-9]+$ && "$REPEATS" =~ ^[1-9][0-9]*$ ]] || {
  echo "INC_SWEEP_WARMUP must be non-negative and INC_SWEEP_REPEATS positive" >&2
  exit 2
}
# Large-message gates use a 1-GiB resident train.  This is large enough to
# amortize persistent-epoch close/rearm costs while still bounding memory for
# the 8-GiB logical case.  It also matches the independently qualified
# transport-roofline measurement scale.
# Keep memory bounded independently of logical volume.  The single-INC relay
# owns ingress plus egress/intermediate storage, so a 1-GiB resident slice can
# exceed device capacity for otherwise legal shapes.  Large logical messages
# are a train of identical 256-MiB epochs and retain the exact total-byte/time
# accounting below.  Platforms may lower this bound explicitly.
MAX_WORKSET=${INC_SWEEP_MAX_WORKSET:-$((256 * 1024 * 1024))}
K1_MAX_WORKSET=${INC_SWEEP_K1_MAX_WORKSET:-$((2 * 1024 * 1024 * 1024))}
REPRESENTATIVE_K1_ROW=${INC_SWEEP_COMBINE_ROW_BYTES_K1:-0}
REPRESENTATIVE_GENERAL_ROW=${INC_SWEEP_COMBINE_ROW_BYTES_GENERAL:-0}
for value in "$REPRESENTATIVE_K1_ROW" "$REPRESENTATIVE_GENERAL_ROW"; do
  [[ "$value" =~ ^[0-9]+$ ]] || exit 2
  (( value == 0 || value % 64 == 0 )) || exit 2
done

align64() { echo $(( ($1 + 63) / 64 * 64 )); }

plan_dispatch() {
  # The sweep volume is the real INC->worker traffic, not logical expert
  # assignments.  With multiple experts on one worker, K can exceed W but a
  # token is still transmitted to that worker only once.  Size the resident
  # train by the physical fanout R=min(K,W); the binary log remains the final
  # source of truth for the report numerator and gate.
  local target=$1 W=$2 K=$3 work=$target row=8192 tokens R=$K
  (( R > W )) && R=$W
  (( work > MAX_WORKSET )) && work=$MAX_WORKSET
  if (( work < W * R * row )); then
    row=$(align64 $(( (work + W * R - 1) / (W * R) )))
    (( row < 64 )) && row=64
    tokens=1
  else
    tokens=$(( (work + W * R * row - 1) / (W * R * row) ))
  fi
  echo "$tokens $row $((W * tokens * R * row))"
}

plan_combine() {
  # A 64-KiB row amortizes per-result ready/RMA control while the 256-MiB
  # bounded working set still supplies many rows to every runtime AIV owner.
  # Tiny totals automatically shrink the row below.
  local target=$1 W=$2 K=$3 work=$target row=65536 results
  # A qualification run may select one regular matrix row for K1 and one for
  # the general reducer.  These are explicit test-shape parameters, not AIV
  # decisions and not hidden W/K tables in the operator.
  if (( K == 1 && REPRESENTATIVE_K1_ROW != 0 )); then
    row=$REPRESENTATIVE_K1_ROW
  elif (( K > 1 && REPRESENTATIVE_GENERAL_ROW != 0 )); then
    row=$REPRESENTATIVE_GENERAL_ROW
  fi
  # Exact one-contribution results use the strict single-INC relay fast path.
  # Keep its resident rows at the transport-qualified 16-KiB transaction
  # point for every worker count.  This is protocol selection by result
  # multiplicity, not a W2/W4/W8 lookup; K>=2 retains the general reducer
  # sizing below and the INC always launches the same fixed AIV service.
  (( K == 1 && REPRESENTATIVE_K1_ROW == 0 )) && row=16384
  # K2 exposes the transport/reduce overlap most directly. Sparse and dense
  # source trains use a 24-KiB row; the medium-density source-local stream
  # benefits from a 64-KiB row. This follows the same runtime density metric
  # as the backend rather than naming a worker-count special case.
  if (( K == 2 && REPRESENTATIVE_GENERAL_ROW == 0 )); then
    local owners=24 owners_per_source
    owners_per_source=$(( (owners + W - 1) / W ))
    if (( owners_per_source >= 4 && owners_per_source < 8 )); then
      row=65536
    else
      row=24576
    fi
  fi
  # Short trains need more independently ready rows to hide fill/drain. Use
  # topology/fanout classes instead of worker-count lookup tables: a narrow
  # fanout on a medium-density source train prefers 32 KiB, while a full
  # fanout on a dense train prefers 24 KiB. Larger resident trains retain the
  # 64-KiB amortization point qualified above.
  if (( K > 1 && work <= 256 * 1024 * 1024 &&
        REPRESENTATIVE_GENERAL_ROW == 0 )); then
    local density_owners=24 density_per_source
    density_per_source=$(( (density_owners + W - 1) / W ))
    if (( density_per_source >= 4 && density_per_source < 8 &&
          K * 2 <= W )); then
      row=32768
    elif (( density_per_source <= 3 && K >= W )); then
      row=24576
    fi
  fi
  local workset_cap=$MAX_WORKSET
  # Pure K1 is a strict equal-rate INC relay.  It needs no reduction scratch,
  # so use the larger live-validated resident train to amortize generation
  # close/rearm.  This consumes INC storage only; K>=2 keeps the conservative
  # general-combine cap.
  (( K == 1 )) && workset_cap=$K1_MAX_WORKSET
  (( work > workset_cap )) && work=$workset_cap
  if (( work < K * row )); then
    row=$(align64 $(( (work + K - 1) / K )))
    (( row < 64 )) && row=64
    results=1
  else
    results=$(( (work + K * row - 1) / (K * row) ))
  fi
  echo "$results $((row / 2)) $((results * K * row))"
}

run_dispatch_case() {
  local W=$1 K=$2 target=$3 dir=$4
  local tokens row op_bytes epochs measure rc=0
  read -r tokens row op_bytes < <(plan_dispatch "$target" "$W" "$K")
  epochs=$(( (target + op_bytes - 1) / op_bytes ))
  measure=$((epochs * REPEATS))
  mkdir -p "$dir"
  rm -f "$dir/shared_log_dir.txt"
  printf 'operator=dispatch workers=%s topk=%s target_bytes=%s op_bytes=%s epochs_per_repeat=%s repeats=%s warmup=%s measured_epochs=%s tokens=%s hidden_bytes=%s\n' \
    "$W" "$K" "$target" "$op_bytes" "$epochs" "$REPEATS" "$WARMUP" "$measure" "$tokens" "$row" >"$dir/meta.txt"
  # The production binary enforces the unique single-INC route for every
  # shape.  The report parser below independently rejects any contrary log.
  INC_STREAM_WARMUP="$WARMUP" INC_STREAM_MEASURE="$measure" \
    "$SCRIPT_DIR/run_single_inc_stream_dispatch_case.sh" \
      "$W" "$tokens" "$row" "$K" "$dir" 120 sweep \
      >"$dir/launcher.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$dir/process_rc.txt"
  echo "SWEEP_CASE operator=dispatch W=$W K=$K target=$target rc=$rc dir=$dir"
}

run_combine_case() {
  local W=$1 K=$2 target=$3 dir=$4
  local results hidden op_bytes epochs measured_epochs rc=0
  read -r results hidden op_bytes < <(plan_combine "$target" "$W" "$K")
  epochs=$(( (target + op_bytes - 1) / op_bytes ))
  measured_epochs=$((epochs * REPEATS))
  mkdir -p "$dir"
  rm -f "$dir/shared_log_dir.txt"
  printf 'operator=combine workers=%s topk=%s target_bytes=%s op_bytes=%s epochs_per_repeat=%s repeats=%s warmup=%s measured_epochs=%s results=%s hidden=%s\n' \
    "$W" "$K" "$target" "$op_bytes" "$epochs" "$REPEATS" "$WARMUP" "$measured_epochs" "$results" "$hidden" >"$dir/meta.txt"
  # K1 is a complete single-INC case too; the production binary makes its
  # identity worker-to-worker shortcut unreachable.
  INC_DYNCSR_SERVICE_EPOCHS="$measured_epochs" \
    INC_DYNCSR_SERVICE_WARMUP_EPOCHS="$WARMUP" \
    "$SCRIPT_DIR/run_single_inc_dyn_case.sh" \
      "$W" "$K" "$results" "$hidden" 0 "$dir" 120 \
      >"$dir/launcher.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$dir/process_rc.txt"
  echo "SWEEP_CASE operator=combine W=$W K=$K target=$target rc=$rc dir=$dir"
}

probe_gate() {
  local operator=$1 dir=$2
  python3 - "$operator" "$dir" <<'PY'
import glob,os,re,sys
op,path=sys.argv[1:]
meta=dict(re.findall(r'(\w+)=([^ ]+)',open(path+'/meta.txt').read()))
W=int(meta['workers']); K=int(meta['topk'])
scale=float(os.environ.get(f'INC_SWEEP_GATE_SCALE_W{W}','1'))
rc=int(open(path+'/process_rc.txt').read().strip())
logical=(int(meta['op_bytes'])*int(meta['epochs_per_repeat'])*
         int(meta['repeats']))
times=[]; sample_max={}
texts=[]
physical_per_iteration=None
input_per_iteration=None
physical_total=None
transport_path=None
for f in glob.glob(path+'/pe*.log'):
    text=open(f,errors='replace').read(); texts.append(text)
    if op=='dispatch':
        mcfg=re.search(r'STREAM_DISPATCH_CONFIG[^\n]* physical_output_bytes=([0-9]+)',text)
        minput=re.search(r'STREAM_DISPATCH_CONFIG[^\n]* input_bytes=([0-9]+)',text)
        mdirect=re.search(r'STREAM_DISPATCH_CONFIG[^\n]* worker_direct=([01])',text)
        if mcfg: physical_per_iteration=int(mcfg.group(1))
        if minput: input_per_iteration=int(minput.group(1))
        if mdirect:
            path_value=('worker_direct_aggregate' if mdirect.group(1)=='1'
                        else 'single_inc_relay')
            if transport_path is not None and transport_path != path_value:
                transport_path='mixed'
            else:
                transport_path=path_value
    else:
        mt=re.search(r'DYNCSR_TIMING[^\n]* physical_ingress_bytes=([0-9]+)',text)
        if mt: physical_total=int(mt.group(1))
    key='STREAM_DISPATCH_TIMING' if op=='dispatch' else 'DYNCSR_TIMING'
    for line in text.splitlines():
        if key not in line: continue
        m=re.search(r'rank_us=([0-9.]+)',line)
        if not m: continue
        t=float(m.group(1)); times.append(t)
        sm=re.search(r'sample=(\d+)',line)
        if sm: sample_max[int(sm.group(1))]=max(sample_max.get(int(sm.group(1)),0.0),t)
slow=(sum(sample_max.values()) if op=='dispatch' else (max(times) if times else 0.0))
physical=((physical_per_iteration or 0)*len(sample_max)
          if op=='dispatch' else (physical_total or 0))
gbps=physical/slow/1000.0 if slow and physical else 0.0
if op == 'dispatch':
    # Dispatch is a full-duplex relay.  Use the measured physical fanout,
    # rather than top-k, because K may exceed W and repeated expert choices on
    # one rank do not add another worker->INC input copy.  This gate is the
    # qualified 93% envelope of C(R)=140.12/(1+0.127/R).
    fanout=((physical_per_iteration or 0)/(input_per_iteration or 1))
    roof=float(os.environ.get('INC_SWEEP_DISPATCH_ROOFLINE','140.12'))
    threshold=0.93*roof*scale/(1.0+0.127/max(fanout,1.0))
else:
    k1=os.environ.get(f'INC_SWEEP_COMBINE_K1_GATE_W{W}')
    threshold=(float(k1) if K==1 and k1 is not None else 120.0*scale)
print(f'SWEEP_PROBE operator={op} GBps={gbps:.3f} dir={path}')
expected=int(meta['workers'])+1
passed=sum(('STREAM_DISPATCH_RESULT' in t and 'pass=1' in t)
           if op=='dispatch' else ('DYNCSR_RESULT SUCCESS' in t)
           for t in texts)
raise SystemExit(0 if rc == 0 and passed == expected and
                 (op != 'dispatch' or transport_path == 'single_inc_relay') and
                 len(texts) == expected and gbps >= threshold else 1)
PY
}

alias_large_case() {
  local operator=$1 W=$2 K=$3 target=$4 source=$5 dir
  dir="$LOG_ROOT/$operator/W${W}/K${K}/B${target}"
  mkdir -p "$dir"
  if [[ "$operator" == dispatch ]]; then
    local tokens row op_bytes iterations
    read -r tokens row op_bytes < <(plan_dispatch "$target" "$W" "$K")
    iterations=$(( (target + op_bytes - 1) / op_bytes ))
    printf 'operator=dispatch workers=%s topk=%s target_bytes=%s op_bytes=%s epochs_per_repeat=%s repeats=1 warmup=0 measured_epochs=%s tokens=%s hidden_bytes=%s shared_large_run=1\n' \
      "$W" "$K" "$target" "$op_bytes" "$iterations" "$iterations" "$tokens" "$row" >"$dir/meta.txt"
  else
    local results hidden op_bytes iterations
    read -r results hidden op_bytes < <(plan_combine "$target" "$W" "$K")
    iterations=$(( (target + op_bytes - 1) / op_bytes ))
    printf 'operator=combine workers=%s topk=%s target_bytes=%s op_bytes=%s epochs_per_repeat=%s repeats=1 warmup=0 measured_epochs=%s results=%s hidden=%s shared_large_run=1\n' \
      "$W" "$K" "$target" "$op_bytes" "$iterations" "$iterations" "$results" "$hidden" >"$dir/meta.txt"
  fi
  printf '%s\n' "$source" >"$dir/shared_log_dir.txt"
  cp "$source/process_rc.txt" "$dir/process_rc.txt"
}

if [[ "${INC_SWEEP_REPORT_ONLY:-0}" != 1 ]]; then
for operator in dispatch combine; do
  if [[ "$operator" == dispatch &&
        "${INC_SWEEP_REUSE_DISPATCH:-0}" == 1 ]]; then
    echo "SWEEP_REUSE operator=dispatch root=$LOG_ROOT"
    continue
  fi
  if [[ -n "${INC_SWEEP_FORCE_OPERATOR:-}" &&
        "$operator" != "${INC_SWEEP_FORCE_OPERATOR}" ]]; then
    echo "SWEEP_SKIP operator=$operator force_operator=${INC_SWEEP_FORCE_OPERATOR}"
    continue
  fi
  for W in "${WORKERS[@]}"; do
    for K in "${TOPKS[@]}"; do
      if [[ "$LEGAL_TOPK_ONLY" == 1 ]] && (( K > W )); then
        echo "SWEEP_SKIP_ILLEGAL_TOPK W=$W K=$K"
        continue
      fi
      for target in "${VOLUMES[@]}"; do
        dir="$LOG_ROOT/$operator/W${W}/K${K}/B${target}"
        force_key="$operator/W${W}/K${K}"
        if [[ "${INC_SWEEP_RESUME:-0}" == 1 &&
              "${INC_SWEEP_FORCE_OPERATOR:-}" != "$operator" &&
              "${INC_SWEEP_FORCE_CASE:-}" != "$force_key" &&
              ! -f "$dir/shared_log_dir.txt" &&
              -f "$dir/process_rc.txt" &&
              "$(cat "$dir/process_rc.txt")" == 0 ]]; then
          if [[ "${INC_SWEEP_RESUME_ALL_SUCCESS:-0}" == 1 ]]; then
            echo "SWEEP_RESUME_CASE operator=$operator W=$W K=$K target=$target"
            continue
          elif (( target == 8589934592 )); then
            if probe_gate "$operator" "$dir"; then
              echo "SWEEP_RESUME_CASE operator=$operator W=$W K=$K target=$target"
              continue
            fi
          else
            echo "SWEEP_RESUME_CASE operator=$operator W=$W K=$K target=$target"
            continue
          fi
        fi
        if [[ "$operator" == dispatch ]]; then
          run_dispatch_case "$W" "$K" "$target" "$dir"
        else
          run_combine_case "$W" "$K" "$target" "$dir"
        fi
        if (( target == 8589934592 )); then
          probe_gate "$operator" "$dir" || {
            echo "SWEEP_PROBE_FAILED operator=$operator W=$W K=$K" >&2
            if [[ "${INC_SWEEP_FAIL_FAST:-0}" == 1 ]]; then
              exit 90
            fi
          }
        fi
      done
    done
  done
done
fi

operator_count=2
if [[ -n "${INC_SWEEP_FORCE_OPERATOR:-}" ]]; then
  if [[ "${INC_SWEEP_FORCE_OPERATOR}" != dispatch &&
        "${INC_SWEEP_FORCE_OPERATOR}" != combine ]]; then
    echo "INC_SWEEP_FORCE_OPERATOR must be dispatch or combine" >&2
    exit 2
  fi
  operator_count=1
fi

shape_count=0
for W in "${WORKERS[@]}"; do
  for K in "${TOPKS[@]}"; do
    if [[ "$LEGAL_TOPK_ONLY" == 1 ]] && (( K > W )); then
      continue
    fi
    shape_count=$((shape_count + 1))
  done
done

python3 - "$LOG_ROOT" "$REPORT" \
  "$((operator_count * shape_count * ${#VOLUMES[@]}))" <<'PY'
import json, os, re, statistics, sys
from pathlib import Path

root, report = Path(sys.argv[1]), Path(sys.argv[2])
expected_case_count = int(sys.argv[3])
dispatch_roofline = float(os.environ.get('INC_SWEEP_DISPATCH_ROOFLINE', '140.12'))
gate_min_bytes = int(os.environ.get('INC_SWEEP_GATE_MIN_BYTES', str(256*1024*1024)))

def worker_gate_scale(workers):
    return float(os.environ.get(f'INC_SWEEP_GATE_SCALE_W{workers}', '1'))

def combine_k1_gate(workers, scale):
    explicit = os.environ.get(f'INC_SWEEP_COMBINE_K1_GATE_W{workers}')
    return float(explicit) if explicit is not None else 120.0 * scale

rows=[]
for meta_path in sorted(root.glob('*/W*/K*/B*/meta.txt')):
    meta=dict(re.findall(r'(\w+)=([^ ]+)',meta_path.read_text()))
    d=meta_path.parent
    source=d
    shared=d/'shared_log_dir.txt'
    if shared.exists(): source=Path(shared.read_text().strip())
    op=meta['operator']; W=int(meta['workers']); K=int(meta['topk'])
    target=int(meta['target_bytes']); op_bytes=int(meta['op_bytes'])
    epochs_per_repeat=int(meta['epochs_per_repeat'])
    repeats=int(meta['repeats'])
    warmup=int(meta['warmup'])
    measured_epochs=int(meta['measured_epochs'])
    rc=int((d/'process_rc.txt').read_text().strip())
    logs=list(source.glob('pe*.log'))
    expected=W+1
    if op=='dispatch':
        passed=sum('STREAM_DISPATCH_RESULT' in t and 'pass=1' in t
                   for t in (p.read_text(errors='replace') for p in logs))
        sample_max={}
        for p in logs:
            for line in p.read_text(errors='replace').splitlines():
                if 'STREAM_DISPATCH_TIMING' not in line: continue
                sm=re.search(r'sample=(\d+)',line); tm=re.search(r'rank_us=([0-9.]+)',line)
                if sm and tm:
                    s=int(sm.group(1)); sample_max[s]=max(sample_max.get(s,0.0),float(tm.group(1)))
        times=[sample_max[k] for k in sorted(sample_max)][:measured_epochs]
        timed_iterations=len(times)
        total_us=sum(times)
        physical_values=[]
        input_values=[]
        direct_values=[]
        for p in logs:
            config_text=p.read_text(errors='replace')
            m=re.search(r'STREAM_DISPATCH_CONFIG[^\n]* physical_output_bytes=([0-9]+)',
                        config_text)
            if m: physical_values.append(int(m.group(1)))
            mi=re.search(r'STREAM_DISPATCH_CONFIG[^\n]* input_bytes=([0-9]+)',
                         config_text)
            if mi: input_values.append(int(mi.group(1)))
            md=re.search(r'STREAM_DISPATCH_CONFIG[^\n]* worker_direct=([01])',
                         config_text)
            if md: direct_values.append(int(md.group(1)))
        physical_per_iteration=(physical_values[0] if physical_values and
                                len(set(physical_values))==1 else 0)
        input_per_iteration=(input_values[0] if input_values and
                             len(set(input_values))==1 else 0)
        physical_bytes=physical_per_iteration*timed_iterations
        repeat_times=[sum(times[i*epochs_per_repeat:(i+1)*epochs_per_repeat])
                      for i in range(repeats)
                      if len(times[i*epochs_per_repeat:(i+1)*epochs_per_repeat]) ==
                         epochs_per_repeat]
        physical_per_repeat=physical_per_iteration*epochs_per_repeat
        repeat_gbps=[physical_per_repeat/t/1000.0 for t in repeat_times if t > 0]
        transport_path=('single_inc_relay' if direct_values and
                        set(direct_values)=={0} else
                        'worker_direct_aggregate' if direct_values and
                        set(direct_values)=={1} else 'unknown')
    else:
        texts=[p.read_text(errors='replace') for p in logs]
        passed=sum('DYNCSR_RESULT SUCCESS' in t for t in texts)
        times=[]
        for t in texts:
            m=re.search(r'DYNCSR_TIMING[^\n]* rank_us=([0-9.]+)',t)
            if m: times.append(float(m.group(1)))
        timed_iterations=measured_epochs
        source_meta=dict(re.findall(r'(\w+)=([^ ]+)',(source/'meta.txt').read_text()))
        source_epochs=int(source_meta['measured_epochs'])
        total_us=(max(times)*measured_epochs/source_epochs) if times else 0.0
        physical_values=[]
        for t in texts:
            m=re.search(r'DYNCSR_TIMING[^\n]* physical_ingress_bytes=([0-9]+)',t)
            if m: physical_values.append(int(m.group(1)))
        source_physical=(physical_values[0] if physical_values and
                         len(set(physical_values))==1 else 0)
        physical_bytes=source_physical*measured_epochs//source_epochs
        physical_per_repeat=physical_bytes//repeats
        repeat_times=[total_us/repeats] * repeats if total_us else []
        # Combine's persistent service reports one device interval spanning all
        # measured epochs.  Its repeat mean is exact; per-repeat variance is
        # intentionally unavailable rather than fabricated from host clocks.
        repeat_gbps=[physical_bytes/total_us/1000.0] * repeats if total_us else []
        direct_values=[]
        for t in texts:
            m=re.search(r'DYNCSR_EVIDENCE[^\n]* k1_direct_result_tx=([01])',t)
            if m: direct_values.append(int(m.group(1)))
        transport_path=('single_inc_combine' if direct_values and
                        set(direct_values)=={0} else
                        'worker_direct_identity' if direct_values and
                        set(direct_values)=={1} else 'unknown')
    logical_bytes=op_bytes*timed_iterations
    gbps=physical_bytes/total_us/1000.0 if total_us else 0.0
    semantic_gbps=logical_bytes/total_us/1000.0 if total_us else 0.0
    correct=(rc==0 and passed==expected and len(logs)==expected and
             ((op=='dispatch' and transport_path=='single_inc_relay') or
              (op=='combine' and transport_path=='single_inc_combine')))
    # The hardware profile chooses the formal large-message boundary. Smaller
    # points remain in the curve and must be correct, but their fixed
    # generation/fill/drain overhead is reported rather than misclassified.
    gated=target>=gate_min_bytes
    if op == 'dispatch':
        physical_fanout=(physical_per_iteration/input_per_iteration
                         if input_per_iteration else 0.0)
        gate_scale=worker_gate_scale(W)
        gate_GBps=(0.93*dispatch_roofline*gate_scale/
                   (1.0+0.127/max(physical_fanout,1.0))
                   if physical_fanout else 0.0)
    else:
        physical_fanout=None
        gate_scale=worker_gate_scale(W)
        gate_GBps=(combine_k1_gate(W, gate_scale)
                   if K==1 else 120.0*gate_scale)
    perf_pass=(not gated or gbps>=gate_GBps)
    rows.append({
        'operator':op,'workers':W,'topk':K,'target_bytes':target,
        'working_set_bytes':op_bytes,
        'epochs_per_repeat':epochs_per_repeat,'repeats':repeats,
        'warmup_epochs':warmup,'measured_epochs':measured_epochs,
        'timed_iterations':timed_iterations,'logical_bytes':logical_bytes,
        'physical_direction_bytes':physical_bytes,
        'physical_direction_bytes_per_repeat':physical_per_repeat,
        'transport_path':transport_path,
        'physical_fanout':physical_fanout,
        'compression_ratio':logical_bytes/physical_bytes if physical_bytes else None,
        'slow_rank_total_us':total_us,
        'mean_repeat_us':statistics.mean(repeat_times) if repeat_times else None,
        'GBps':gbps,
        'mean_repeat_GBps':statistics.mean(repeat_gbps) if repeat_gbps else None,
        'repeat_GBps_min':min(repeat_gbps) if repeat_gbps else None,
        'repeat_GBps_max':max(repeat_gbps) if repeat_gbps else None,
        'repeat_GBps_stddev':(
            statistics.pstdev(repeat_gbps)
            if op=='dispatch' and len(repeat_gbps)>1 else None),
        'repeat_GBps_cv':(
            statistics.pstdev(repeat_gbps)/statistics.mean(repeat_gbps)
            if op=='dispatch' and len(repeat_gbps)>1 and
               statistics.mean(repeat_gbps)>0 else None),
        'repeat_variance_source':(
            'per_epoch_device_intervals_grouped_by_logical_repeat'
            if op=='dispatch' else 'aggregate_persistent_device_interval'),
        'semantic_GBps':semantic_gbps,'correct':correct,
        'performance_gated':gated,'performance_gate_GBps':gate_GBps,
        'performance_pass':perf_pass,
        'process_rc':rc,'passed_ranks':passed,'expected_ranks':expected,
        'log_dir':str(d),'shared_log_dir':str(source) if source != d else None})
failures=[r for r in rows if not r['correct'] or not r['performance_pass']]
correctness_failures=[r for r in rows if not r['correct']]
performance_failures=[r for r in rows if not r['performance_pass']]
rows.sort(key=lambda r:(r['operator'],r['workers'],r['topk'],r['target_bytes']))
sizes=sorted({r['target_bytes'] for r in rows})
topks=sorted({r['topk'] for r in rows})
workers=sorted({r['workers'] for r in rows})
doc={
  'schema':'single_inc_operator_dense_sweep_v5_profiled_gate',
  'sizes_bytes':sizes,
  'topk':topks, 'workers':workers,
  'warmup_epochs':sorted({r['warmup_epochs'] for r in rows}),
  'repeats':sorted({r['repeats'] for r in rows}),
  'repeat_method':(
      'one process per case; warmup epochs precede measured epochs; each '
      'logical repeat covers target_bytes rounded up to whole resident epochs'),
  'bandwidth_numerator':'dispatch physical_output_bytes; combine physical_ingress_bytes',
  'large_message_target_GBps':None,
  'gate_min_bytes':gate_min_bytes,
  'gate_policy':{
      'dispatch':('target>=256MiB: 0.93 * dispatch_roofline * '
                  'worker_gate_scale / (1 + 0.127/R), '
                  'R=physical_output_bytes/input_bytes'),
      'dispatch_roofline_GBps':dispatch_roofline,
      'worker_gate_scale':{str(w):worker_gate_scale(w) for w in workers},
      'combine_K1_relay_GBps':{
          str(w):combine_k1_gate(w,worker_gate_scale(w)) for w in workers},
      'combine_Kge2_base_GBps':120.0},
  'expected_case_count':expected_case_count,
  'case_count':len(rows),'failure_count':len(failures),
  'correctness_failure_count':len(correctness_failures),
  'performance_failure_count':len(performance_failures),
  'correctness_pass':len(rows)==expected_case_count and not correctness_failures,
  'rows':rows,'failures':failures,
  'pass':len(rows)==expected_case_count and not failures}
report.parent.mkdir(parents=True,exist_ok=True)
report.write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps({'case_count':len(rows),'failure_count':len(failures),'pass':doc['pass'],'report':str(report)}))
if not doc['pass']: raise SystemExit(1)
PY
