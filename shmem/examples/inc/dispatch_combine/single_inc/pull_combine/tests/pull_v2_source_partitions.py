#!/usr/bin/env python3
"""Device qualification for independent origin partitions and real D->C."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from pull_v2_k4_formal import HELPERS, wait_selected_devices_idle


def run_case(args, name, workers, rows, hidden, topk, mode='chain',
             route='random_gpu', seed=17, measure=3, warmup=0, delay=0):
    wait_selected_devices_idle(args.first_npu // 8 * 8, 8, args.timeout)
    case = args.output / name
    endpoint = f'tcp://127.0.0.1:{HELPERS.free_port()}'
    tail = [','.join(map(str, rows)), str(hidden), '64', str(topk),
            str(seed), str(measure), str(delay), mode, str(warmup), route]
    commands = [[str(workers), str(pe), endpoint, str(args.first_npu), *tail]
                for pe in range(workers + 1)]
    procs, logs = [], []
    try:
        procs, logs = HELPERS.start_group(
            args.build_dir / 'bin/inc_dc_source_partition_device_e2e',
            commands, case / 'gate', 'partition', case / 'logs',
            args.build_dir, args.timeout)
        deadline = time.monotonic() + args.timeout
        for proc in procs:
            proc.wait(timeout=max(.1, deadline - time.monotonic()))
        for log in logs:
            log.flush()
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError(f'{name}: exits {[p.returncode for p in procs]}')
        for pe in range(workers + 1):
            if '[PASS]' not in (case / 'logs' / f'pe{pe}.log').read_text():
                raise RuntimeError(f'{name}: PE{pe} missing PASS')
        records = [json.loads(line) for line in
                   (case / 'logs' / f'pe{workers}.log').read_text().splitlines()
                   if line.startswith('{')]
        samples = [r for r in records if r['test'] == 'source_partition_device_e2e']
        if len(samples) != warmup + measure or not all(r['correct'] for r in samples):
            raise RuntimeError(f'{name}: sample count/correctness')
        if delay:
            first = samples[0]
            if not first.get('early_independence_checked') or not first.get('early_independence_passed'):
                raise RuntimeError(f'{name}: missing independent early completion proof')
        summary = next(r for r in records if r['test'] == 'source_partition_device_e2e_summary')
        if mode != 'chain' and not all(k in summary for k in (
                'directional_min_gb_s', 'directional_mean_gb_s', 'directional_cv_percent')):
            raise RuntimeError(f'{name}: missing directional performance metrics')
        result = dict(case=name, workers=workers, first_npu=args.first_npu,
                      rows=','.join(map(str, rows)), hidden=hidden, topk=topk,
                      mode=mode, route=route, seed=seed, warmup=warmup,
                      measure=measure, delay_ms=delay, correct=True,
                      minimum_gb_s=summary.get('directional_min_gb_s', ''),
                      mean_gb_s=summary.get('directional_mean_gb_s', ''),
                      cv_percent=summary.get('directional_cv_percent', ''),
                      mean_us=summary['mean_us'])
        print(json.dumps(result), flush=True)
        return result
    finally:
        HELPERS.terminate(procs)
        for log in logs:
            log.close()
        wait_selected_devices_idle(args.first_npu // 8 * 8, 8, args.timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--first-npu', type=int, default=0)
    p.add_argument('--suite', choices=('early', 'smoke', 'performance', 'random-stress', 'random-combine', 'combine-regression'), default='early')
    p.add_argument('--timeout', type=float, default=300)
    a = p.parse_args()
    if a.first_npu < 0 or a.first_npu + 4 >= 16 or a.first_npu // 8 != (a.first_npu + 4) // 8:
        p.error('five NPUs must fit in one HCCS plane')
    a.output.mkdir(parents=True, exist_ok=False)
    cases = []
    if a.suite == 'early':
        for w in (2, 4):
            cases.append((f'w{w}_late0', w, [32] * w, 64, 2, dict(delay=100)))
        cases.append(('ragged_tail', 4, [31, 0, 7, 19], 33, 4, dict(delay=100)))
    elif a.suite == 'smoke':
        for w in (2, 4):
            cases.append((f'w{w}_empty', w, [0] * w, 3, 2, {}))
            cases.append((f'w{w}_random', w, [64] * w, 8192, 2, dict(route='random_expert')))
        cases.append(('ragged', 4, [31, 0, 7, 19], 33, 4, {}))
    elif a.suite == 'combine-regression':
        for w, k, regular in ((2, 2, 'sym_k2_balanced'),
                               (4, 2, 'sym_k2_balanced'),
                               (4, 4, 'sym_k4_gpu4')):
            for route in (regular, 'random_expert'):
                cases.append((f'combine_w{w}_k{k}_{route}', w, [8192] * w,
                              8192, k, dict(mode='combine', route=route,
                                            warmup=3, measure=10)))
        for h in (3, 2047, 2049, 8193):
            cases.append((f'chain_h{h}', 4, [31, 0, 7, 19], h, 4,
                          dict(route='random_expert', measure=10, delay=100)))
        cases.append(('all_empty', 4, [0] * 4, 33, 4, dict(measure=10)))
    elif a.suite in ('random-stress', 'random-combine'):
        # Fixed hidden input scale; Combine ingress is measured from actual
        # distinct-GPU contributions, not expert top-k times input bytes.
        for w, k in ((2, 2), (4, 2), (4, 4)):
            for seed in (17, 101, 2027):
                for mode in (('combine',) if a.suite == 'random-combine'
                             else ('dispatch', 'combine')):
                    cases.append((f'{mode}_w{w}_k{k}_seed{seed}', w,
                                  [8192] * w, 8192, k,
                                  dict(mode=mode, route='random_expert',
                                       seed=seed, warmup=3, measure=10)))
        cases.append(('random_chain_ragged', 4, [8192, 0, 4097, 1023],
                      8192, 4, dict(route='random_expert', measure=10)))
    else:
        for w, k, route in ((2, 2, 'sym_k2_balanced'),
                            (4, 2, 'sym_k2_balanced'),
                            (4, 4, 'sym_k4_gpu4')):
            for mode in ('dispatch', 'combine'):
                rows = 8192 if mode == 'dispatch' else (128 << 20) // (k * 8192 * 4)
                cases.append((f'{mode}_w{w}_k{k}', w, [rows] * w, 8192, k,
                              dict(mode=mode, route=route, warmup=3, measure=10)))
    with (a.output / 'summary.csv').open('x', newline='') as stream:
        writer = None
        for name, w, rows, h, k, options in cases:
            result = run_case(a, name, w, rows, h, k, **options)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            stream.flush()


if __name__ == '__main__':
    main()
