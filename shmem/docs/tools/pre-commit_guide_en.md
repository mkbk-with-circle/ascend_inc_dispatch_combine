# Usage Guide of pre-commit for Code Check

## Overview

This project uses the [pre-commit](https://pre-commit.com/) framework to automatically check code quality before code is committed, ensuring consistent code styling and preventing common errors.

## Installation

### 1. Installing pre-commit

```bash
pip install pre-commit
```

### 2. (Recommended) Installing Git Hooks

```bash
pre-commit install
```

After installation, every time you run `git commit`, the configured checks will be executed automatically.

## Instructions

### Automatic Check (Recommended)

Git hooks installed trigger checks automatically when you commit code.

```bash
git add .
git commit -m "your message"
```

If the check fails, tools like ruff-format and clang-format can fix the issues automatically. After the issues are fixed, commit the code again.

### Manual Check

Check the files temporarily stored:

```bash
pre-commit run
```

Check specified files:

```bash
pre-commit run ruff-check --files path/to/file.py
pre-commit run clang-format --files path/to/file.cpp
```

Check a single hook:

```bash
pre-commit run ruff-check
pre-commit run pylint
pre-commit run clang-format
```

### Check Skip (Not Recommended)

```bash
git commit --no-verify -m "your message"
```

## Check Tools

| Tool| Language| Function| Configuration File|
|------|------|------|----------|
| ruff | Python | Code formatting + Lint| `tools/pre-commit/pyproject.toml` |
| pylint | Python | Code quality check| `tools/pre-commit/pyproject.toml` |
| bandit | Python | Security vulnerability check| `tools/pre-commit/pyproject.toml` |
| codespell | General| Spell check| `.pre-commit-config.yaml` |
| typos | General| Spell check| `tools/pre-commit/typos.toml` |
| clang-format | C/C++ | Code formatting| `.clang-format` |

## Configuration Files

### Main Configuration File

[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml): check tools and parameters to be executed

### Python Tool Configuration

[`tools/pre-commit/pyproject.toml`](../../tools/pre-commit/pyproject.toml): Ruff, Pylint, and Bandit rules

### C++ Formatting Configuration

[`.clang-format`](../../.clang-format): formatting rules used by clang-format

### Spell Check Whitelist

[`tools/pre-commit/typos.toml`](../../tools/pre-commit/typos.toml): false positive whitelist of typos

## FAQs

### Question: What Can I Do If the Check Fails?

Tools like ruff-format and clang-format can fix issues automatically. You can commit the code again directly. For issues that need to be manually fixed, modify the code based on the error message and commit the code again.

### Question: How Do I Update Pre-commit Hooks?

```bash
pre-commit autoupdate
```

### Question: How Do I View Detailed Error Information About a Tool?

```bash
pre-commit run pylint --verbose
```

### Question: How Do I Temporarily Disable a Rule?

**Python (Ruff/Pylint):** Add comments at the end of a code line.

```python
x = 1  # pylint: disable=invalid-name
```

**C++ (clang-format):** Use comments to enclose the code.

```cpp
// clang-format off
int unformatted_code = 1;
// clang-format on
```

### Question: What Can I Do If the First Run Is Slow?

On the first run, you need to download and install each check tool's environment. After that, the cache speeds up the process.

## Best Practices

1. **Install Git hooks**: Use `pre-commit install` to automatically run checks on every commit.
2. **Avoid frequent `--no-verify`**: Skipping checks may allow problematic code to enter the repository.
3. **Keep hooks updated**: Regularly run `pre-commit autoupdate` to fetch the latest versions of hooks.
4. **Configure IDE integration**: Configure the ruff and clang-format plugins in your IDE for real-time code checks.
