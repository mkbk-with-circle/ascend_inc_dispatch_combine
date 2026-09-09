#!/usr/bin/env python3
"""Full-operator random routing qualification, with auditable payload sizes."""
import argparse
import csv
import json
from pathlib import Path

from pull_v2_dispatch_unified import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--first-npu', default=0, type=int)
    p.add_argument('--workers', nargs='+', default=[2, 4], type=int, choices=[2, 4])
    p.add_argument('--seeds', nargs='+', default=[17, 104729, 20260909], type=int)
    p.add_argument('--routes', nargs='+', default=[
        'random_k2_gpu2', 'random_k4_gpu2', 'random_k4_gpu4', 'random_k8_gpu4',
        'random_expert_k2', 'random_expert_k4', 'random_expert_k8'])
    p.add_argument('--payload', default=128 << 20, type=int)
    p.add_argument('--hidden', default=8192, type=int)
    p.add_argument('--warmup', default=3, type=int)
    p.add_argument('--measure', default=10, type=int)
    p.add_argument('--timeout', default=120, type=float)
    a = p.parse_args()
    if (a.hidden <= 0 or a.payload < 0 or a.warmup < 0 or a.measure <= 0 or
            a.first_npu < 0 or a.first_npu + max(a.workers) >= 16 or
            a.first_npu // 8 != (a.first_npu + max(a.workers)) // 8):
        p.error('invalid shape/count or ranks outside one nb HCCS plane')
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / 'summary.csv').open('x', newline='') as stream:
        writer = None
        for workers in a.workers:
            for route in a.routes:
                if workers == 2 and route in ('random_k4_gpu4', 'random_k8_gpu4'):
                    continue
                for seed in a.seeds:
                    name = f'w{workers}_{route}_s{seed}'
                    result = run(a, name, a.build_dir, workers, a.hidden,
                                 a.payload, route, a.warmup, a.measure, seed=seed)
                    log = a.output / name / 'logs' / f'pe{workers}.log'
                    expected = a.payload // (a.hidden * 2) * (a.hidden * 2)
                    for line in log.read_text().splitlines():
                        if not line.startswith('{'):
                            continue
                        row = json.loads(line)
                        if row.get('test') == 'pull_dispatch_v2_device_e2e':
                            if row['source_hidden_bytes'] != [expected] * workers:
                                raise RuntimeError(f'{name}: source payload mismatch')
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(result))
                        writer.writeheader()
                    writer.writerow(result)
                    stream.flush()


if __name__ == '__main__':
    main()
