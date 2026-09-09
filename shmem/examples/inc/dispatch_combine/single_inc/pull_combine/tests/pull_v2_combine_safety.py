#!/usr/bin/env python3
"""Bounded device regression for fixed2/fixed4 validation and ring reuse."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def run_case(args, case):
    name, route, hidden, rows, repeats, fault, aiv = case
    folder = args.output / name
    folder.mkdir()
    for device in range(args.first_npu, args.first_npu + 5):
        state = subprocess.run(
            ['npu-smi', 'info', '-t', 'proc-mem', '-i', str(device), '-c', '0'],
            capture_output=True, text=True, check=True)
        if 'No process in device.' not in state.stdout:
            raise RuntimeError(f'NPU {device} is occupied')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        endpoint = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f'{args.build_dir}/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['SHMEM_UID_SESSION_ID'] = f'combine-safety-{os.getpid()}-{time.monotonic_ns()}'
    env.pop('INC_DC_PULL_V2_START_GATE_DIR', None)
    env.pop('INC_DC_PULL_V2_COMBINE_ACTIVE_AIV', None)
    if aiv:
        env['INC_DC_PULL_V2_COMBINE_ACTIVE_AIV'] = str(aiv)
    binary = args.build_dir / 'bin/inc_dc_pull_combine_v2_npu_e2e'
    procs, logs = [], []
    try:
        for pe in range(5):
            log = (folder / f'pe{pe}.log').open('w')
            logs.append(log)
            procs.append(subprocess.Popen([
                str(binary), '4', str(pe), endpoint, str(args.first_npu),
                str(hidden), str(rows), route, '0', str(repeats), str(fault)
            ], env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout
        for proc in procs:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        if any(p.returncode for p in procs):
            raise RuntimeError(f'{name}: rank exits {[p.returncode for p in procs]}')
        for log in logs:
            log.flush()
        records = [json.loads(line) for line in
                   (folder / 'pe4.log').read_text().splitlines()
                   if line.startswith('{')]
        samples = [r for r in records if r['test'] == 'pull_combine_v2_npu_e2e']
        if len(samples) != repeats or not all(
                r['correct'] and r['status'] == (2 if fault else 0)
                for r in samples):
            raise RuntimeError(f'{name}: sample/status verification failed')
        print(f'PASS {name}: {repeats} waves', flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-npu', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if args.first_npu < 0 or args.first_npu + 4 >= 16 or (
            args.first_npu // 8 != (args.first_npu + 4) // 8):
        parser.error('nb safety qualification requires five NPUs in one HCCS plane')
    args.output.mkdir(parents=True, exist_ok=False)
    for route in ('sym_k2_balanced', 'sym_k4_gpu4'):
        for fault in range(1, 6):
            run_case(args, (f'{route}_fault{fault}', route, 2049, 32, 3, fault, 0))
        run_case(args, (f'{route}_reuse', route, 2049, 32, 100, 0, 0))
        run_case(args, (f'{route}_one_aiv', route, 8192, 32, 3, 0, 1))
    run_case(args, ('empty', 'asymmetric', 3, 0, 3, 0, 0))
    run_case(args, ('short_tail', 'sym_k4_gpu4', 3, 7, 3, 0, 0))
    for route in ('fixed_k1', 'fixed_k3', 'mixed_k', 'asymmetric'):
        run_case(args, (route + '_tail', route, 2049, 257, 5, 0, 0))


if __name__ == '__main__':
    main()
