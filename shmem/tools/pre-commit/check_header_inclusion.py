#!/usr/bin/env python3
"""
Pre-commit hook: Check explicit header inclusion in C/C++ files.

Uses clang-tidy's misc-include-cleaner (AST-based semantic analysis) to check:
1. Standard library header: symbols must have their corresponding headers included.
2. Header self-contained: .h/.hpp files should include headers for all
   project-internal types/macros they use.
3. Unused includes: headers included but not directly used.

Usage:
    python tools/pre-commit/check_header_inclusion.py [files ...]
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# clang-tidy binary detection
# ---------------------------------------------------------------------------

_CLANG_TIDY_PATHS = [
    # pip-installed (this project's standard path)
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), '.venv', 'bin', 'clang-tidy'),
    '/tmp/pip_pkgs/clang_tidy/data/bin/clang-tidy',
    '/tmp/pip_pkgs/bin/clang-tidy',
    # system
    'clang-tidy',
    'clang-tidy-18', 'clang-tidy-17', 'clang-tidy-16',
    'clang-tidy-15', 'clang-tidy-14', 'clang-tidy-22', 'clang-tidy-20',
    '/usr/bin/clang-tidy',
    '/usr/local/bin/clang-tidy',
]

# ---------------------------------------------------------------------------
# Include path auto-detection
# ---------------------------------------------------------------------------

def _auto_detect_include_paths(project_root: str) -> List[str]:
    """Auto-detect include directories from project structure.

    Recursively scans source trees for directories containing C/C++ headers,
    so clang-tidy can resolve all #include directives without a build.
    """
    paths: List[str] = []
    root_dirs = ['include', 'src', 'examples']
    for root_dir in root_dirs:
        root = os.path.join(project_root, root_dir)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if any(f.endswith(('.h', '.hpp', '.hxx')) for f in filenames):
                paths.append(dirpath)
    return paths


def _candidate_ascend_roots() -> List[str]:
    """Return common Ascend toolkit roots that may provide kernel_operator.h."""
    roots: List[str] = []
    for env_name in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME'):
        env_value = os.environ.get(env_name)
        if env_value:
            roots.append(env_value)

    home = os.path.expanduser('~')
    roots.extend([
        '/usr/local/Ascend/ascend-toolkit/latest',
        '/usr/local/Ascend/latest',
        os.path.join(home, 'Ascend', 'ascend-toolkit', 'latest'),
    ])

    developer_ascend = os.path.join(home, 'Ascend')
    if os.path.isdir(developer_ascend):
        for name in os.listdir(developer_ascend):
            path = os.path.join(developer_ascend, name)
            if os.path.isdir(path):
                roots.append(path)

    seen: Set[str] = set()
    result: List[str] = []
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if root not in seen and os.path.isdir(root):
            seen.add(root)
            result.append(root)
    return result


def _auto_detect_ascend_include_paths() -> List[str]:
    """Auto-detect AscendC include directories for device headers."""
    search_roots = [
        'include',
        'asc/include',
        'ascendc/include',
        'include/ascendc',
        'aarch64-linux/include',
        'aarch64-linux/asc/include',
        'aarch64-linux/asc/impl',
        'aarch64-linux/ascendc/include',
        'aarch64-linux/include/ascendc',
    ]

    paths: List[str] = []
    seen: Set[str] = set()
    for root in _candidate_ascend_roots():
        for rel_path in search_roots:
            search_root = os.path.join(root, rel_path)
            if not os.path.isdir(search_root):
                continue
            for dirpath, _, filenames in os.walk(search_root):
                if not any(f.endswith(('.h', '.hpp', '.hxx')) for f in filenames):
                    continue
                if dirpath not in seen:
                    seen.add(dirpath)
                    paths.append(dirpath)
    return paths


def _find_compile_commands(project_root: str) -> Optional[str]:
    """Find compile_commands.json in common build locations."""
    candidates = [
        os.path.join(project_root, 'build', 'compile_commands.json'),
        os.path.join(project_root, 'compile_commands.json'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None




def _extract_extra_args(compile_commands_path: str, files: List[str],
                         project_root: str) -> List[str]:
    """Extract portable compiler flags (-I, -D, -std=) from compile_commands.json."""
    try:
        with open(compile_commands_path, 'r') as f:
            db = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    target_files: Set[str] = set()
    for f in files:
        target_files.add(os.path.abspath(os.path.join(project_root, f)))

    seen_args: Set[str] = set()
    result: List[str] = []

    for entry in db:
        entry_file = entry.get('file', '')
        if entry_file not in target_files:
            continue

        cmd = entry.get('command', '')
        parts = cmd.split()

        for i, part in enumerate(parts):
            if part == '-I' and i + 1 < len(parts):
                arg = '-I' + parts[i + 1]
                if arg not in seen_args:
                    seen_args.add(arg)
                    result.append(arg)
            elif part.startswith('-I'):
                if part not in seen_args:
                    seen_args.add(part)
                    result.append(part)
            elif part == '-isystem' and i + 1 < len(parts):
                arg = '-isystem' + parts[i + 1]
                if arg not in seen_args:
                    seen_args.add(arg)
                    result.append(arg)
            elif part.startswith('-isystem'):
                if part not in seen_args:
                    seen_args.add(part)
                    result.append(part)
            elif part.startswith('-D') or part.startswith('-std='):
                if part not in seen_args:
                    seen_args.add(part)
                    result.append(part)

    if not result:
        result.extend(
            '-I' + p for p in _auto_detect_include_paths(project_root)
        )
        result.extend(
            '-I' + p for p in _auto_detect_ascend_include_paths()
        )
        result.append('-std=c++17')

    return result

# ---------------------------------------------------------------------------
# clang-tidy wrapper
# ---------------------------------------------------------------------------

def _find_clang_tidy() -> Optional[str]:
    """Find the clang-tidy binary."""
    # 1. Known absolute paths
    for p in _CLANG_TIDY_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    # 2. Pre-commit / pip venv: use the clang_tidy Python module
    try:
        import clang_tidy  # type: ignore
        exe = clang_tidy.get_executable("clang-tidy")
        if exe and os.path.isfile(str(exe)) and os.access(str(exe), os.X_OK):
            return str(exe)
    except (ImportError, AttributeError, FileNotFoundError):
        pass

    # 3. PATH lookup
    for p in ['clang-tidy', 'clang-tidy-18', 'clang-tidy-17', 'clang-tidy-16',
              'clang-tidy-15', 'clang-tidy-14', 'clang-tidy-22', 'clang-tidy-20']:
        try:
            result = subprocess.run([p, '--version'], capture_output=True,
                                    timeout=5)
            if result.returncode == 0:
                return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _run_clang_tidy(clang_tidy: str, files: List[str], project_root: str) -> str:
    """Run clang-tidy misc-include-cleaner on given files.

    Returns (stdout+stderr, returncode).
    """
    args = [clang_tidy, '--checks=-*,misc-include-cleaner']

    compile_commands = _find_compile_commands(project_root)
    if compile_commands:
        extra_args = _extract_extra_args(compile_commands, files, project_root)
        for ea in extra_args:
            args.extend(['-extra-arg=' + ea])
    else:
        include_paths = _auto_detect_include_paths(project_root)
        for p in include_paths:
            args.extend(['-extra-arg=-I' + p])
        for p in _auto_detect_ascend_include_paths():
            args.extend(['-extra-arg=-I' + p])
        args.extend(['-extra-arg-before=-xc++'])
        args.extend(['-extra-arg=-std=c++17'])

    args.extend(files)

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=project_root,
    )
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_CHECK_TYPE_HEADER_REGEX = re.compile(
    r'no header providing "(.*?)" is directly included'
)
_UNUSED_INCLUDE_REGEX = re.compile(
    r'included header (.*?) is not used directly'
)
_WARNING_LINE_REGEX = re.compile(
    r'^(.+?):(\d+):(\d+):\s+warning:\s+(.+?)\s+\[misc-include-cleaner\]'
)
_DIAGNOSTIC_ERROR_REGEX = re.compile(r'\[clang-diagnostic-error\]')


def parse_clang_tidy_output(output: str) -> Tuple[List[Dict], bool]:
    """Parse clang-tidy output.  Returns (violations, has_compilation_error)."""
    violations: List[Dict] = []
    has_compilation_error = False

    for line in output.splitlines():
        if _DIAGNOSTIC_ERROR_REGEX.search(line):
            has_compilation_error = True
            continue

        m = _WARNING_LINE_REGEX.match(line.strip())
        if not m:
            continue

        file_path = m.group(1)
        line_num = int(m.group(2))
        message = m.group(4)

        header_match = _CHECK_TYPE_HEADER_REGEX.search(message)
        unused_match = _UNUSED_INCLUDE_REGEX.search(message)

        if header_match:
            symbol = header_match.group(1)
            if symbol.startswith('std::') or symbol in (
                    'malloc', 'free', 'calloc', 'realloc', 'memset', 'memcpy',
                    'memmove', 'memcmp', 'strlen', 'strcmp', 'strncmp',
                    'strcpy', 'strncpy', 'strcat', 'strncat', 'strchr',
                    'strrchr', 'strstr', 'strtok', 'printf', 'fprintf',
                    'sprintf', 'snprintf', 'puts', 'fputs', 'fopen', 'fclose',
                    'fread', 'fwrite', 'fseek', 'ftell', 'fgets', 'scanf',
                    'sscanf', 'assert', 'errno', 'abs', 'sqrt', 'pow', 'ceil',
                    'floor', 'round', 'log', 'exp',
            ):
                check_type = 'stdlib_include'
                formatted = f"missing include for {symbol} - add the corresponding header"
            else:
                check_type = 'header_self'
                formatted = f"no header providing \"{symbol}\" is directly included"
            violations.append({
                'file': file_path,
                'line': line_num,
                'message': formatted,
                'check_type': check_type,
            })
        elif unused_match:
            header = unused_match.group(1)
            violations.append({
                'file': file_path,
                'line': line_num,
                'message': f"included header {header} is not used directly",
                'check_type': 'unused_include',
            })
        else:
            violations.append({
                'file': file_path,
                'line': line_num,
                'message': message,
                'check_type': 'unknown',
            })

    return violations, has_compilation_error


def _extract_error_lines(output: str) -> List[str]:
    """Extract compilation error lines for user diagnosis."""
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _DIAGNOSTIC_ERROR_REGEX.search(stripped):
            lines.append(stripped)
        elif 'fatal error:' in stripped:
            lines.append(stripped)
        elif 'Found compiler error' in stripped:
            lines.append(stripped)
    return lines


def _is_missing_toolchain_header(error_lines: List[str]) -> bool:
    """Detect parse failures caused by missing external toolchain headers."""
    missing_header_re = re.compile(
        r"(?:fatal )?error: '([^']+)' file not found"
    )
    toolchain_headers = {
        'cstdint',
        'stdint.h',
        'kernel_operator.h',
        'kernel_tpipe.h',
        'kernel_macros.h',
    }
    for line in error_lines:
        match = missing_header_re.search(line)
        if not match:
            continue
        header = match.group(1)
        if header in toolchain_headers or header.startswith(('kernel_', 'type_')):
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check explicit header inclusion in C/C++ files '
                    '(powered by clang-tidy misc-include-cleaner).'
    )
    parser.add_argument(
        'files', nargs='*',
        help='Files to check (pre-commit passes staged files)'
    )
    parser.add_argument(
        '--project-root', default='.',
        help='Project root directory (default: current directory)'
    )
    parser.add_argument(
        '--check', default='all',
        choices=['stdlib', 'header_self', 'unused', 'all'],
        help='Which checks to run (default: all)'
    )
    parser.add_argument(
        '--output', default='text',
        choices=['text', 'json'],
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--exclude', nargs='*', default=[],
        help='File glob patterns to exclude'
    )
    return parser.parse_args()


_CHECK_LABELS: Dict[str, str] = {
    'stdlib_include': 'Missing standard library headers',
    'header_self': 'Header self-contained issues',
    'unused_include': 'Unused includes',
    'unknown': 'Other include issues',
}


def _report_text(violations: List[Dict]) -> str:
    """Format violations as text output."""
    if not violations:
        return ''

    groups: Dict[str, List[Dict]] = {}
    for v in violations:
        ct = v['check_type']
        groups.setdefault(ct, []).append(v)

    lines: List[str] = []
    for ct in ('stdlib_include', 'header_self', 'unused_include', 'unknown'):
        if ct in groups:
            label = _CHECK_LABELS.get(ct, ct)
            lines.append(f"=== {label} ===")
            for v in groups[ct]:
                if v['line'] > 0:
                    lines.append(f"{v['file']}:{v['line']}: {v['message']}")
                else:
                    lines.append(f"{v['file']}: {v['message']}")
            lines.append('')

    lines.append(f"Total: {len(violations)} violation(s) found.")
    return '\n'.join(lines)


def _filter_by_check_type(violations: List[Dict], check: str) -> List[Dict]:
    """Filter violations based on requested check type."""
    if check == 'all':
        return violations
    type_map = {
        'stdlib': 'stdlib_include',
        'header_self': 'header_self',
        'unused': 'unused_include',
    }
    target = type_map.get(check, check)
    return [v for v in violations if v['check_type'] == target]


def main() -> int:
    args = _parse_args()

    project_root = os.path.abspath(args.project_root)

    c_extensions = {'.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cxx', '.hxx'}
    files = []
    for f in args.files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in c_extensions:
            continue
        if not os.path.isabs(f):
            f = os.path.join(project_root, f)
        f = os.path.relpath(f, project_root)
        excluded = False
        for pattern in args.exclude:
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(os.path.basename(f), pattern):
                excluded = True
                break
        if not excluded and os.path.isfile(os.path.join(project_root, f)):
            files.append(f)

    if not files:
        return 0

    clang_tidy = _find_clang_tidy()
    if not clang_tidy:
        print("ERROR: clang-tidy not found. "
              "Install with: pip install clang-tidy --target=/tmp/pip_pkgs",
              file=sys.stderr)
        return 2

    try:
        output = _run_clang_tidy(clang_tidy, files, project_root)
    except subprocess.TimeoutExpired:
        print("ERROR: clang-tidy timed out (>120s). Consider checking fewer files.",
              file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"ERROR: clang-tidy binary not found at '{clang_tidy}'.",
              file=sys.stderr)
        return 2

    violations, has_compilation_error = parse_clang_tidy_output(output)

    # If clang-tidy had compilation errors AND produced zero include warnings,
    # the results are unreliable — the source likely cannot be parsed at all.
    if has_compilation_error and not violations:
        error_lines = _extract_error_lines(output)
        if _is_missing_toolchain_header(error_lines):
            print("SKIP: clang-tidy could not parse this file because "
                  "required C++/AscendC toolchain headers are unavailable "
                  "in the current environment.")
            for el in error_lines[:10]:
                print(f"  {el}")
            return 0
        print("ERROR: clang-tidy failed to check this file — compilation "
              "errors prevent include analysis.\n"
              "Fix the errors below or ensure all include paths are available:\n",
              file=sys.stderr)
        for el in error_lines[:20]:
            print(f"  {el}", file=sys.stderr)
        if len(error_lines) > 20:
            print(f"  ... and {len(error_lines) - 20} more error(s)",
                  file=sys.stderr)
        return 2

    violations = _filter_by_check_type(violations, args.check)

    if violations:
        report = (_report_text(violations) if args.output == 'text'
                  else json.dumps(violations, indent=2))
        print(report, file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
