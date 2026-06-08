# Contributing to UAVidToolKit

Thank you for taking the time to contribute. The following guidelines help keep the codebase consistent and make the review process as smooth as possible.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Report a Bug](#how-to-report-a-bug)
3. [How to Request a Feature](#how-to-request-a-feature)
4. [Development Setup](#development-setup)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Coding Standards](#coding-standards)
7. [Commit Message Format](#commit-message-format)

---

## Code of Conduct

This project follows a simple standard: be respectful and constructive. Harassment, personal attacks, and discriminatory language are not acceptable in any project space.

---

## How to Report a Bug

Before opening a new issue, please search the [issue tracker](https://github.com/dronefreak/UAVidToolKit/issues) to check whether it has already been reported.

When filing a bug report, include:

- **Environment** — operating system, Python version, and the output of `pip list` for relevant packages.
- **Steps to reproduce** — a minimal, self-contained example that demonstrates the problem.
- **Expected behaviour** — what you expected to happen.
- **Actual behaviour** — what actually happened, including the full traceback if applicable.

---

## How to Request a Feature

Open an issue with the title prefix `[Feature]` and describe:

- The problem or limitation you are experiencing.
- The proposed solution or behaviour you would like to see.
- Any alternatives you have considered.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/UAVidToolKit.git
cd UAVidToolKit

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build the Cython extension (optional, required for evaluation tests)
python setup.py build_ext --inplace

# 5. Create a feature branch
git checkout -b feature/your-feature-name
```

---

## Submitting a Pull Request

1. **Keep changes focused.** One logical change per pull request makes review faster and rollback easier.
2. **Write or update tests.** All new behaviour should be covered by tests.
3. **Ensure tests pass** before opening the PR.
4. **Update documentation.** If your change affects public-facing behaviour, update the relevant docstrings and `README.md`.
5. **Open the PR against `main`.** Fill in the pull-request template, describing what changed and why.

A maintainer will review your PR and may request changes before merging.

---

## Coding Standards

- **Style** — Follow [PEP 8](https://peps.python.org/pep-0008/). Maximum line length is 100 characters.
- **Docstrings** — Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions, methods, and classes.
- **Type hints** — Encouraged for new code, though not required for modifications to existing functions.
- **Cython** — Changes to `.pyx` files must be rebuilt (`python setup.py build_ext --inplace`) and verified on a 64-bit Linux system before submission.

---

## Commit Message Format

Use the conventional commits format:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Common types:

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (dependency updates, build changes) |

**Examples:**

```
feat(evaluate): add per-class GT pixel count reporting

fix(colorTransformer): use base-256 hash to prevent RGB collisions

docs(readme): add label colour map table and argument reference tables
```
