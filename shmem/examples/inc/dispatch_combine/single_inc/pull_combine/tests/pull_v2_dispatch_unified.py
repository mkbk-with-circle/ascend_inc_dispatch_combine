#!/usr/bin/env python3
"""Complete Dispatch qualification; optional same-plane baseline comparison."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from pull_v2_k4_formal import HELPERS, wait_iteration, wait_selected_devices_idle


def run(args, name, build, workers, hidden, payload, route, warmup, measure,
        fault=0, token_skew=0, ready_skew=0):
    plane_start = args.first_npu // 8 * 8
    wait_selected_devices_idle(plane_start, 8, args.timeout)
    out = args.output / name
    endpoint = f'tcp://127.0.0.1:{HELPERS.free_port()}'
    commands = [[str(workers), str(pe), endpoint, str(args.first_npu), str(payload),
                 route, str(hidden), '64', '5' if workers == 2 else '3',
                 str(warmup), str(measure), '20260909', str(fault)]
                for pe in range(workers + 1)]
    if token_skew or ready_skew:
        for command in commands:
            command.extend([str(token_skew), str(ready_skew)])
    procs, logs = [], []
    try:
        procs, logs = HELPERS.start_group(
            build / 'bin/inc_dc_pull_dispatch_v2_device_e2e', commands,
            out / 'gate', 'dispatch', out / 'logs', build, args.timeout)
        cpu_groups = getattr(args, 'cpu_groups', [])
        if cpu_groups:
            for pe, proc in enumerate(procs):
                os.sched_setaffinity(proc.pid, cpu_groups[pe])
        for iteration in range(warmup + measure):
            wait_iteration(out / 'gate', 'dispatch', iteration, workers, procs, args.timeout)
        deadline = time.monotonic() + args.timeout
        for proc in procs:
            proc.wait(timeout=max(.1, deadline - time.monotonic()))
        if any(p.returncode for p in procs):
            raise RuntimeError(f'{name}: exits {[p.returncode for p in procs]}')
        for log in logs:
            log.flush()
        for pe in range(workers + 1):
            if '[PASS]' not in (out / 'logs' / f'pe{pe}.log').read_text():
                raise RuntimeError(f'{name}: rank {pe} failed')
        records = [json.loads(line) for line in
                   (out / 'logs' / f'pe{workers}.log').read_text().splitlines()
                   if line.startswith('{')]
        samples = [r for r in records if r['test'] == 'pull_dispatch_v2_device_e2e']
        expected = [0, 8, 10, 13, 3, 12][fault]
        if len(samples) != warmup + measure or not all(
                r['correct'] and r['status'] == expected for r in samples):
            raise RuntimeError(f'{name}: incorrect sample/status count')
        measured = [r['downlink_gb_s'] for r in samples if r['phase'] == 'measure']
        mean = statistics.mean(measured)
        result = dict(case=name, workers=workers, hidden=hidden, payload=payload,
                      route=route, fault=fault, samples=measure, correct=True,
                      token_skew=token_skew, ready_skew_us=ready_skew,
                      minimum=min(measured) if not fault else None,
                      mean=mean if not fault else None,
                      cv=(100 * statistics.pstdev(measured) / mean if mean else 0)
                      if not fault else None)
        print(json.dumps(result), flush=True)
        return result
    finally:
        try:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for proc in procs:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
        finally:
            for log in logs:
                log.close()
            wait_selected_devices_idle(plane_start, 8, args.timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-npu', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--mode', choices=('safety', 'performance'), default='safety')
    parser.add_argument('--cpu-groups', default='',
                        help='optional five rank CPU sets, e.g. 0,1:2,3:4,5:6,7:48,49')
    args = parser.parse_args()
    args.cpu_groups = [set(map(int, part.split(',')))
                       for part in args.cpu_groups.split(':') if part]
    if args.cpu_groups and (len(args.cpu_groups) != 5 or any(
            not group <= os.sched_getaffinity(0) for group in args.cpu_groups)):
        parser.error('--cpu-groups requires five allowed CPU sets')
    if args.first_npu < 0 or args.first_npu + 4 >= 16 or (
            args.first_npu // 8 != (args.first_npu + 4) // 8):
        parser.error('nb qualification requires five NPUs in one plane')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    regressions = []
    if args.mode == 'performance':
        if not args.baseline_dir:
            parser.error('--baseline-dir is required for a comparison')
        # A baseline may contain only its kernel .so; use the candidate's
        # identical harness/transport dependencies for both measurements.
        os.environ['LD_LIBRARY_PATH'] = str(args.build_dir / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        for workers, route in [(2, 'sym_k2_balanced'), (4, 'sym_k2_balanced'),
                               (4, 'sym_k4_gpu4'), (4, 'sym_k4_gpu2'), (4, 'sym_k8_gpu4')]:
            pair = []
            for label, build in [('baseline', args.baseline_dir), ('candidate', args.build_dir)]:
                row = run(args, f'w{workers}_{route}_{label}', build, workers,
                          8192, 128 << 20, route, 3, 10)
                pair.append(row)
                results.append(row)
            mean_ratio = pair[1]['mean'] / pair[0]['mean']
            min_ratio = pair[1]['minimum'] / pair[0]['minimum']
            print(f'COMPARE W{workers} {route}: mean={mean_ratio:.6f} min={min_ratio:.6f}', flush=True)
            if mean_ratio <= .99 or min_ratio <= .99:
                regressions.append(f'W{workers}/{route}')
    else:
        ordinal = 0
        for workers in (2, 4):
            cases = [(8192, 0, 'ragged', 3), (3, 48, 'ragged', 5),
                     (3, 48, 'sym_k2_balanced', 5), (33, 16384, 'ragged', 5),
                     (32, 16384, 'ragged', 5), (8192, 1048576, 'ragged', 5),
                     (8192, 1048576, 'hotspot', 5),
                     (8192, 4096, 'sym_k2_balanced', 100)]
            for hidden, payload, route, repeats in cases:
                results.append(run(args, f'safety_{ordinal}', args.build_dir, workers,
                                   hidden, payload, route, 0, repeats))
                ordinal += 1
        for fault in range(1, 6):
            results.append(run(args, f'fault_{fault}', args.build_dir, 4,
                               8192, 1048576, 'sym_k4_gpu4', 0, 3, fault))
        for workers in (2, 4):
            results.append(run(args, f'skew_w{workers}', args.build_dir, workers,
                               8192, 1048576, 'ragged', 0, 3,
                               token_skew=75, ready_skew=1000))
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    if regressions:
        raise RuntimeError('performance decrease is not below 1%: ' + ', '.join(regressions))


if __name__ == '__main__':
    main()
