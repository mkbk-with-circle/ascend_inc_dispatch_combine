#!/usr/bin/env python3
"""解析 INC Fusion Kernel 的聚合设备 trace。

本工具故意只依赖 Python 标准库。当前设备日志没有逐 token-wave 的绝对
timestamp，因此这里只计算日志能够证明的 D/C 聚合服务窗口，不推导或绘制伪造的
逐 wave timeline。
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
AGGREGATION_NOTICE = (
    "当前 FUSION_* trace 和 INC_FUSION_DEVICE_TRACE records 是整次 kernel/角色的"
    "聚合跨度及相对 checkpoint，不含每个 token wave 的绝对起止 timestamp；"
    "因此可以计算 D/C 聚合窗口收益，但不能据此还原或伪造逐 wave timeline。"
)

MARKERS = (
    "FUSION_INC_DISPATCH_CHECKPOINTS",
    "FUSION_INC_COMBINE_CHECKPOINTS",
    "FUSION_WORKER_TRACE",
    "FUSION_INC_TRACE",
    "INC_FUSION_DEVICE_TRACE",
)
MARKER_RE = re.compile("|".join(re.escape(marker) for marker in MARKERS))
NUMBER_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
ROLE_RE = re.compile(r"\brole(?P<role>\d+)=(?P<cycles>\d+)")
DEVICE_HEAD_RE = re.compile(
    r"INC_FUSION_DEVICE_TRACE\s+"
    r"rank=(?P<rank>\d+)\s+layer=(?P<layer>\d+)\s+"
    r"call=(?P<call>\d+)\s+records\s*=\s*"
)

ROLE_NAMES = {
    1: "INC Dispatch",
    2: "INC Combine",
    3: "Worker Dispatch 上传",
    4: "Worker Dispatch 接收/打包",
    5: "Worker FFN 计算",
    6: "Worker Combine",
}

CSV_FIELDS = (
    "case",
    "source",
    "line",
    "event_type",
    "aggregation",
    "aggregation_notice",
    "pe",
    "rank",
    "layer",
    "call",
    "role",
    "role_name",
    "lane",
    "cycles",
    "checkpoints",
    "d_span_cycles",
    "c_span_cycles",
    "overlap_cycles",
    "serial_cycles",
    "concurrent_window_cycles",
    "overlap_realized",
    "theoretical_max_speedup",
    "actual_window_speedup",
    "actual_vs_theoretical_pct",
    "d_span_us",
    "c_span_us",
    "overlap_us",
    "concurrent_window_us",
    "trace_valid",
)


def _role_name(role: int) -> str:
    return ROLE_NAMES.get(role, "未知角色")


def _segments(line: str, marker: str) -> Iterable[Tuple[int, str]]:
    """返回某 marker 到下一个已知 marker 之间的片段。"""
    all_markers = list(MARKER_RE.finditer(line))
    for index, match in enumerate(all_markers):
        if match.group(0) != marker:
            continue
        end = all_markers[index + 1].start() if index + 1 < len(all_markers) else len(line)
        yield match.start(), line[match.end():end].strip()


def _parse_int_list(text: str) -> Optional[List[int]]:
    match = re.search(r"\d+(?:\s*,\s*\d+)*", text)
    if not match:
        return None
    return [int(item.strip()) for item in match.group(0).split(",")]


def _extract_balanced_literal(text: str, start: int) -> Tuple[str, int]:
    """从 start 提取一个考虑字符串转义的 Python list/dict literal。"""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] not in "[{":
        raise ValueError("records= 后不是 list/dict")

    pairs = {"[": "]", "{": "}"}
    stack: List[str] = []
    quote: Optional[str] = None
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "]}":
            if not stack or char != stack[-1]:
                raise ValueError("records literal 括号不匹配")
            stack.pop()
            if not stack:
                return text[start:position + 1], position + 1
    raise ValueError("records literal 未闭合")


def _as_number(value: str) -> Any:
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    return float(value)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def calculate_dc_metrics(
    emitted: Dict[str, Any], clock_mhz: Optional[float] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """重新计算 D/C 窗口指标，返回 metrics 和数据告警。"""
    warnings: List[str] = []
    required = ("d_span_cycles", "c_span_cycles", "overlap_cycles")
    missing = [key for key in required if key not in emitted]
    if missing:
        return {}, ["FUSION_INC_TRACE 缺少字段：" + ", ".join(missing)]

    try:
        d_span = int(emitted["d_span_cycles"])
        c_span = int(emitted["c_span_cycles"])
        overlap = int(emitted["overlap_cycles"])
    except (TypeError, ValueError, OverflowError):
        return {}, ["FUSION_INC_TRACE 的 D/C/overlap 不是有效整数"]

    valid = True
    if min(d_span, c_span, overlap) < 0:
        warnings.append("D/C/overlap 出现负数")
        valid = False
    if overlap > min(d_span, c_span):
        warnings.append("overlap_cycles 大于较短的 D/C span，时间窗口不自洽")
        valid = False

    serial = d_span + c_span
    concurrent = serial - overlap
    if concurrent <= 0:
        warnings.append("D+C-overlap 非正，无法计算实际 window speedup")
        valid = False

    shorter = min(d_span, c_span)
    longer = max(d_span, c_span)
    realized = overlap / shorter if shorter > 0 else None
    theoretical = serial / longer if longer > 0 else None
    actual = serial / concurrent if serial > 0 and concurrent > 0 else None
    efficiency = (
        actual / theoretical * 100.0
        if actual is not None and theoretical not in (None, 0.0)
        else None
    )

    metrics: Dict[str, Any] = {
        "d_span_cycles": d_span,
        "c_span_cycles": c_span,
        "overlap_cycles": overlap,
        "serial_cycles": serial,
        "concurrent_window_cycles": concurrent,
        "overlap_realized": realized,
        "theoretical_max_speedup": theoretical,
        "actual_window_speedup": actual,
        "actual_vs_theoretical_pct": efficiency,
        "trace_valid": valid,
    }
    if clock_mhz is not None:
        metrics.update({
            "d_span_us": d_span / clock_mhz,
            "c_span_us": c_span / clock_mhz,
            "overlap_us": overlap / clock_mhz,
            "serial_us": serial / clock_mhz,
            "concurrent_window_us": concurrent / clock_mhz,
        })

    comparisons = {
        "overlap_realized": realized,
        "dc_theoretical_max": theoretical,
        "dc_window_speedup": actual,
    }
    for emitted_key, calculated in comparisons.items():
        if emitted_key not in emitted or calculated is None:
            continue
        reported = emitted[emitted_key]
        if not _finite_number(reported):
            warnings.append(f"{emitted_key} 不是有限数")
            continue
        if not math.isclose(float(reported), calculated, rel_tol=5e-4, abs_tol=5e-4):
            warnings.append(
                f"{emitted_key}={reported} 与重算值 {calculated:.8g} 不一致"
            )
    return metrics, warnings


def _parse_worker_segment(
    body: str, source: str, line_number: int, warnings: List[str]
) -> Optional[Dict[str, Any]]:
    pe_match = re.search(r"\bpe=(\d+)", body)
    role_matches = list(ROLE_RE.finditer(body))
    if pe_match is None or not role_matches:
        warnings.append(f"{source}:{line_number}: 无法解析 FUSION_WORKER_TRACE")
        return None

    roles = []
    for index, match in enumerate(role_matches):
        end = role_matches[index + 1].start() if index + 1 < len(role_matches) else len(body)
        tail = body[match.end():end]
        checkpoints_match = re.search(r"\bcheckpoints=([0-9]+(?:\s*,\s*[0-9]+)*)", tail)
        role = int(match.group("role"))
        roles.append({
            "role": role,
            "role_name": _role_name(role),
            "cycles": int(match.group("cycles")),
            "checkpoints": (
                [int(value.strip()) for value in checkpoints_match.group(1).split(",")]
                if checkpoints_match else []
            ),
        })
    return {
        "type": "worker_trace",
        "source": source,
        "line": line_number,
        "pe": int(pe_match.group(1)),
        "aggregation": "whole_kernel_role_aggregate",
        "roles": roles,
    }


def _normalise_device_records(
    value: Any, source: str, line_number: int, warnings: List[str]
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        warnings.append(f"{source}:{line_number}: device records 不是 list")
        return None
    records: List[Dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            warnings.append(f"{source}:{line_number}: records[{index}] 不是 dict")
            continue
        required = ("role", "lane", "cycles", "checkpoints")
        if any(key not in raw for key in required):
            warnings.append(
                f"{source}:{line_number}: records[{index}] 缺少 role/lane/cycles/checkpoints"
            )
            continue
        try:
            role = int(raw["role"])
            lane = int(raw["lane"])
            cycles = int(raw["cycles"])
            checkpoints = [int(item) for item in raw["checkpoints"]]
        except (TypeError, ValueError, OverflowError):
            warnings.append(f"{source}:{line_number}: records[{index}] 字段类型无效")
            continue
        record = dict(raw)
        record.update({
            "role": role,
            "role_name": _role_name(role),
            "lane": lane,
            "cycles": cycles,
            "checkpoints": checkpoints,
        })
        records.append(record)
    return records


def parse_line(
    line: str,
    source: str,
    line_number: int,
    clock_mhz: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """解析一行；支持一行拼接多个 vLLM device trace。"""
    events: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for _, body in _segments(line, "FUSION_WORKER_TRACE"):
        event = _parse_worker_segment(body, source, line_number, warnings)
        if event is not None:
            events.append(event)

    checkpoint_types = (
        ("FUSION_INC_DISPATCH_CHECKPOINTS", "inc_dispatch_checkpoints"),
        ("FUSION_INC_COMBINE_CHECKPOINTS", "inc_combine_checkpoints"),
    )
    for marker, event_type in checkpoint_types:
        for _, body in _segments(line, marker):
            checkpoints = _parse_int_list(body)
            if checkpoints is None:
                warnings.append(f"{source}:{line_number}: 无法解析 {marker}")
                continue
            events.append({
                "type": event_type,
                "source": source,
                "line": line_number,
                "aggregation": "lane0_whole_kernel_offsets",
                "checkpoints": checkpoints,
            })

    for _, body in _segments(line, "FUSION_INC_TRACE"):
        emitted = {
            match.group("key"): _as_number(match.group("value"))
            for match in NUMBER_RE.finditer(body)
        }
        metrics, metric_warnings = calculate_dc_metrics(emitted, clock_mhz)
        warnings.extend(f"{source}:{line_number}: {item}" for item in metric_warnings)
        if not emitted:
            warnings.append(f"{source}:{line_number}: 无法解析 FUSION_INC_TRACE")
            continue
        events.append({
            "type": "inc_trace",
            "source": source,
            "line": line_number,
            "aggregation": "whole_kernel_dc_window",
            "emitted": emitted,
            "metrics": metrics,
        })

    search_position = 0
    while True:
        match = DEVICE_HEAD_RE.search(line, search_position)
        if match is None:
            break
        try:
            literal, end = _extract_balanced_literal(line, match.end())
            raw_records = ast.literal_eval(literal)
            records = _normalise_device_records(raw_records, source, line_number, warnings)
            if records is not None:
                events.append({
                    "type": "device_trace",
                    "source": source,
                    "line": line_number,
                    "rank": int(match.group("rank")),
                    "layer": int(match.group("layer")),
                    "call": int(match.group("call")),
                    "aggregation": "whole_kernel_role_aggregate",
                    "records": records,
                })
            search_position = end
        except (ValueError, SyntaxError) as error:
            warnings.append(
                f"{source}:{line_number}: 无法解析 INC_FUSION_DEVICE_TRACE: {error}"
            )
            search_position = match.end()

    if "INC_FUSION_DEVICE_TRACE" in line and not DEVICE_HEAD_RE.search(line):
        warnings.append(f"{source}:{line_number}: device trace 头部格式不完整")
    return events, warnings


def _case_identity(path: Optional[Path], override: Optional[str]) -> Tuple[str, str]:
    if override:
        return override, override
    if path is None:
        return "stdin", "stdin"
    parent = path.resolve().parent
    return str(parent), parent.name or str(parent)


def parse_stream(
    lines: Iterable[str],
    source: str,
    path: Optional[Path],
    report: Dict[str, Any],
    case_name: Optional[str],
    clock_mhz: Optional[float],
) -> None:
    case_key, display_name = _case_identity(path, case_name)
    cases = report.setdefault("_case_map", {})
    case = cases.setdefault(case_key, {
        "name": display_name,
        "directory": case_key,
        "files": [],
        "events": [],
    })
    if source not in case["files"]:
        case["files"].append(source)
    for line_number, line in enumerate(lines, 1):
        events, warnings = parse_line(line, source, line_number, clock_mhz)
        case["events"].extend(events)
        report["warnings"].extend(warnings)


def _expand_inputs(values: Sequence[str]) -> List[Optional[Path]]:
    paths: List[Optional[Path]] = []
    for value in values:
        if value == "-":
            paths.append(None)
            continue
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(value)
        if path.is_dir():
            matches = sorted(
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in {".log", ".txt", ".out"}
            )
            if not matches:
                raise FileNotFoundError(f"目录中没有 .log/.txt/.out：{value}")
            paths.extend(matches)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"不是普通文件：{value}")
    if sum(path is None for path in paths) > 1:
        raise ValueError("标准输入 '-' 最多指定一次")
    return paths


def parse_inputs(
    values: Sequence[str],
    case_name: Optional[str] = None,
    clock_mhz: Optional[float] = None,
) -> Dict[str, Any]:
    if clock_mhz is not None and (not math.isfinite(clock_mhz) or clock_mhz <= 0):
        raise ValueError("--clock-mhz 必须是正的有限数")
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "aggregation_notice": AGGREGATION_NOTICE,
        "clock_mhz": clock_mhz,
        "inputs": [],
        "cases": [],
        "warnings": [],
    }
    paths = _expand_inputs(values)
    for path in paths:
        if path is None:
            source = "<stdin>"
            report["inputs"].append(source)
            parse_stream(sys.stdin, source, None, report, case_name, clock_mhz)
            continue
        resolved = path.resolve()
        source = str(resolved)
        report["inputs"].append(source)
        with resolved.open("r", encoding="utf-8", errors="replace") as handle:
            parse_stream(handle, source, resolved, report, case_name, clock_mhz)

    case_map = report.pop("_case_map", {})
    report["cases"] = sorted(case_map.values(), key=lambda item: item["directory"])
    event_count = sum(len(case["events"]) for case in report["cases"])
    if event_count == 0:
        report["warnings"].append("输入中没有识别到任何 Fusion trace")
    return report


def _short_source(source: str) -> str:
    if source == "<stdin>":
        return source
    return Path(source).name


def _fmt_number(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{float(value):.{digits}f}{suffix}"


def _fmt_cycles(value: Optional[int], metrics: Dict[str, Any], us_key: str) -> str:
    if value is None:
        return "—"
    if us_key in metrics:
        return f"{value} ({metrics[us_key]:.3f} µs)"
    return str(value)


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _event_counts(case: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in case["events"]:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    return counts


def render_markdown(report: Dict[str, Any], detail_limit: int = 200) -> str:
    lines = [
        "# INC Fusion timeline 解析报告",
        "",
        f"> **数据边界：** {report['aggregation_notice']}",
        "",
        "## 输入汇总",
        "",
        "| Case | 文件数 | D/C 窗口 | Worker trace | vLLM device trace |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        counts = _event_counts(case)
        lines.append(
            f"| {_md_escape(case['name'])} | {len(case['files'])} | "
            f"{counts.get('inc_trace', 0)} | {counts.get('worker_trace', 0)} | "
            f"{counts.get('device_trace', 0)} |"
        )
    if not report["cases"]:
        lines.append("| — | 0 | 0 | 0 | 0 |")

    for case in report["cases"]:
        lines.extend(["", f"## Case：`{_md_escape(case['name'])}`", ""])
        traces = [event for event in case["events"] if event["type"] == "inc_trace"]
        lines.extend([
            "### D/C 聚合服务窗口",
            "",
            "理论上限为 `(D+C)/max(D,C)`；实际 window speedup 为 "
            "`(D+C)/(D+C-overlap)`。两者都是 D/C 服务窗口口径，不等于端到端加速比。",
            "",
        ])
        if traces:
            lines.extend([
                "| 来源 | D span | C span | overlap | overlap 实现率 | 并发窗口 | 理论上限 | 实际 window speedup | 达到理论 | 自洽 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            ])
            for event in traces:
                metrics = event.get("metrics", {})
                source = f"{_short_source(event['source'])}:{event['line']}"
                lines.append(
                    f"| {_md_escape(source)} | "
                    f"{_fmt_cycles(metrics.get('d_span_cycles'), metrics, 'd_span_us')} | "
                    f"{_fmt_cycles(metrics.get('c_span_cycles'), metrics, 'c_span_us')} | "
                    f"{_fmt_cycles(metrics.get('overlap_cycles'), metrics, 'overlap_us')} | "
                    f"{_fmt_number(metrics.get('overlap_realized') * 100.0 if metrics.get('overlap_realized') is not None else None, 2, '%')} | "
                    f"{_fmt_cycles(metrics.get('concurrent_window_cycles'), metrics, 'concurrent_window_us')} | "
                    f"{_fmt_number(metrics.get('theoretical_max_speedup'), 4, '×')} | "
                    f"{_fmt_number(metrics.get('actual_window_speedup'), 4, '×')} | "
                    f"{_fmt_number(metrics.get('actual_vs_theoretical_pct'), 2, '%')} | "
                    f"{'是' if metrics.get('trace_valid') else '否'} |"
                )
        else:
            lines.extend([
                "本 case 没有 `FUSION_INC_TRACE`，无法从 worker/device 聚合记录反推 D/C "
                "绝对窗口或交叠收益。",
                "",
            ])

        checkpoint_events = [
            event for event in case["events"]
            if event["type"] in {"inc_dispatch_checkpoints", "inc_combine_checkpoints"}
        ]
        if checkpoint_events:
            lines.extend([
                "",
                "### INC lane-0 聚合 checkpoints",
                "",
                "这些数值是相对各自角色起点的 cycle offset，覆盖整次 kernel；不是逐 wave timestamp。",
                "",
                "| 来源 | 角色 | checkpoint offsets (cycles) |",
                "|---|---|---|",
            ])
            for event in checkpoint_events[:detail_limit or None]:
                name = "Dispatch" if event["type"] == "inc_dispatch_checkpoints" else "Combine"
                lines.append(
                    f"| {_md_escape(_short_source(event['source']))}:{event['line']} | "
                    f"{name} | `{','.join(map(str, event['checkpoints']))}` |"
                )

        worker_rows = []
        for event in case["events"]:
            if event["type"] != "worker_trace":
                continue
            for role in event["roles"]:
                worker_rows.append((event, role))
        if worker_rows:
            lines.extend([
                "",
                "### Worker 角色聚合 trace",
                "",
                "| 来源 | PE | 角色 | span (cycles) | checkpoint offsets |",
                "|---|---:|---|---:|---|",
            ])
            for event, role in worker_rows[:detail_limit or None]:
                checkpoints = ",".join(map(str, role["checkpoints"])) or "—"
                lines.append(
                    f"| {_md_escape(_short_source(event['source']))}:{event['line']} | "
                    f"{event['pe']} | {role['role']} / {_md_escape(role['role_name'])} | "
                    f"{role['cycles']} | `{checkpoints}` |"
                )
            if detail_limit and len(worker_rows) > detail_limit:
                lines.append(f"\n仅展示前 {detail_limit}/{len(worker_rows)} 行；JSON/CSV 保留全部记录。")

        device_rows = []
        for event in case["events"]:
            if event["type"] != "device_trace":
                continue
            for record in event["records"]:
                device_rows.append((event, record))
        if device_rows:
            lines.extend([
                "",
                "### vLLM `INC_FUSION_DEVICE_TRACE` 聚合 records",
                "",
                "| 来源 | rank | layer/call | 角色/lane | span (cycles) | checkpoint offsets |",
                "|---|---:|---|---|---:|---|",
            ])
            for event, record in device_rows[:detail_limit or None]:
                checkpoints = ",".join(map(str, record["checkpoints"])) or "—"
                lines.append(
                    f"| {_md_escape(_short_source(event['source']))}:{event['line']} | "
                    f"{event['rank']} | {event['layer']}/{event['call']} | "
                    f"{record['role']} / {_md_escape(record['role_name'])} / {record['lane']} | "
                    f"{record['cycles']} | `{checkpoints}` |"
                )
            if detail_limit and len(device_rows) > detail_limit:
                lines.append(f"\n仅展示前 {detail_limit}/{len(device_rows)} 行；JSON/CSV 保留全部记录。")

    if report["warnings"]:
        lines.extend(["", "## 解析告警", ""])
        lines.extend(f"- {_md_escape(item)}" for item in report["warnings"])
    lines.extend(["", "---", "", f"数据口径声明：{report['aggregation_notice']}", ""])
    return "\n".join(lines)


def _csv_rows(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for case in report["cases"]:
        for event in case["events"]:
            base = {
                "case": case["name"],
                "source": event["source"],
                "line": event["line"],
                "event_type": event["type"],
                "aggregation": event.get("aggregation", ""),
                "aggregation_notice": report["aggregation_notice"],
            }
            if event["type"] == "inc_trace":
                row = dict(base)
                row.update(event.get("metrics", {}))
                yield row
            elif event["type"] in {"inc_dispatch_checkpoints", "inc_combine_checkpoints"}:
                row = dict(base)
                row["checkpoints"] = ",".join(map(str, event["checkpoints"]))
                yield row
            elif event["type"] == "worker_trace":
                for role in event["roles"]:
                    row = dict(base)
                    row.update({
                        "pe": event["pe"],
                        "role": role["role"],
                        "role_name": role["role_name"],
                        "cycles": role["cycles"],
                        "checkpoints": ",".join(map(str, role["checkpoints"])),
                    })
                    yield row
            elif event["type"] == "device_trace":
                for record in event["records"]:
                    row = dict(base)
                    row.update({
                        "rank": event["rank"],
                        "layer": event["layer"],
                        "call": event["call"],
                        "role": record["role"],
                        "role_name": record["role_name"],
                        "lane": record["lane"],
                        "cycles": record["cycles"],
                        "checkpoints": ",".join(map(str, record["checkpoints"])),
                    })
                    yield row


def render_csv(report: Dict[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in _csv_rows(report):
        writer.writerow(row)
    return output.getvalue()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="解析单 INC Fusion Kernel 的聚合 timeline 日志。",
        epilog=(
            "示例：python3 parse_fusion_timeline.py case/pe*.log -o timeline.md\n"
            "      python3 parse_fusion_timeline.py case --format json -o timeline.json\n\n"
            "注意：当前日志不含逐 token-wave 的绝对 timestamp，本工具不会伪造 wave 级时间线。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="一个或多个日志文件/目录；目录递归读取 .log/.txt/.out；'-' 表示标准输入",
    )
    parser.add_argument(
        "--format", choices=("markdown", "json", "csv"), default="markdown",
        help="输出格式（默认：markdown）",
    )
    parser.add_argument("-o", "--output", help="输出文件；省略时写到标准输出")
    parser.add_argument(
        "--clock-mhz", type=float,
        help="可选的设备 cycle clock（MHz）；仅在已知实测频率时用于换算 µs，不设置默认值",
    )
    parser.add_argument(
        "--case-name",
        help="覆盖 case 名称，并把所有输入合并为一个 case；默认按日志父目录分组",
    )
    parser.add_argument(
        "--detail-limit", type=int, default=200,
        help="Markdown 每类详情最多显示的行数，0 表示不限制（默认：200；JSON/CSV 不截断）",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="存在解析告警时返回非零；适合 CI/资格化脚本",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.detail_limit < 0:
        parser.error("--detail-limit 不能为负数")
    try:
        report = parse_inputs(args.inputs, args.case_name, args.clock_mhz)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    event_count = sum(len(case["events"]) for case in report["cases"])
    if args.format == "json":
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    elif args.format == "csv":
        rendered = render_csv(report)
    else:
        rendered = render_markdown(report, args.detail_limit)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    for warning in report["warnings"]:
        print(f"警告：{warning}", file=sys.stderr)
    if event_count == 0 or (args.strict and report["warnings"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
