#!/usr/bin/env python3
"""Same-plane regression check for READY push followed by Notice."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pull_v2_source_partitions import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=(2, 4), required=True)
    parser.add_argument('--first-npu', type=int, choices=(0, 8), required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for version, build in [('baseline', a.baseline), ('push_ready', a.candidate)]:
        hashes[version] = {
            name: hashlib.sha256((build / name).read_bytes()).hexdigest()
            for name in ('bin/inc_dc_source_partition_device_e2e',
                         'lib/libinc_dc_partitioned_combine.so',
                         'lib/libinc_dc_pull_dispatch_v2_device_kernel.so')}
    assert hashes['baseline']['bin/inc_dc_source_partition_device_e2e'] == hashes['push_ready']['bin/inc_dc_source_partition_device_e2e']
    assert hashes['baseline']['lib/libinc_dc_pull_dispatch_v2_device_kernel.so'] == hashes['push_ready']['lib/libinc_dc_pull_dispatch_v2_device_kernel.so']
    (a.output / 'builds.json').write_text(json.dumps(hashes, indent=2) + '\n')
    run = SimpleNamespace(first_npu=a.first_npu, timeout=600,
                          build_dir=a.candidate, output=a.output)
    with (a.output / 'summary.csv').open('x', newline='') as out:
        writer = None

        def record(name, rows, h, k, **kwargs):
            nonlocal writer
            result = run_case(run, name, a.workers, rows, h, k, **kwargs)
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            out.flush()
            return result

        record('empty', [0] * a.workers, 33, 2, measure=3)
        record('ragged_delayed', [17, 9] if a.workers == 2 else [31, 0, 7, 19],
               33, a.workers, measure=3, delay=100, route='random_expert')
        for k in ([2] if a.workers == 2 else [2, 4]):
            measured = []
            for version, build in [('baseline', a.baseline), ('push_ready', a.candidate)]:
                run.build_dir = build
                measured.append(record(f'{version}_k{k}', [8192] * a.workers,
                                       8192, k, mode='combine', route='random_expert',
                                       seed=17, warmup=3, measure=10))
            old, new = measured
            ratio = new['mean_gb_s'] / old['mean_gb_s']
            minimum_ratio = new['minimum_gb_s'] / old['minimum_gb_s']
            print(json.dumps(dict(workers=a.workers, topk=k, mean_ratio=ratio,
                                  minimum_ratio=minimum_ratio)), flush=True)
            if ratio < .99 or minimum_ratio < .99:
                raise RuntimeError('More than 1% bandwidth regression: inspect preserved logs')


if __name__ == '__main__':
    main()
