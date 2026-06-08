# Security Policy

## Supported Versions

Only the latest commit on the `main` branch receives security fixes. No long-term support versions are maintained at this time.

| Version | Supported |
|---|---|
| `main` (latest) | Yes |
| Older commits | No |

---

## Reporting a Vulnerability

If you discover a security vulnerability, **do not open a public GitHub issue**. Public disclosure before a fix is available puts all users at risk.

Instead, please report it privately by one of the following methods:

1. **GitHub private vulnerability reporting** — Use the [Security tab](https://github.com/dronefreak/UAVidToolKit/security/advisories/new) on the repository to submit a draft security advisory. This keeps the details confidential until a fix is released.

2. **Email** — If you prefer email, contact the maintainer directly through the email address listed on their [GitHub profile](https://github.com/dronefreak).

### What to include in your report

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce the issue, including any relevant code, configuration, or environment details.
- A proposed fix or mitigation, if you have one.

---

## Response Process

1. You will receive an acknowledgement within **5 business days** of submission.
2. The maintainer will investigate and, where necessary, prepare a fix on a private branch.
3. A patched release will be published and the vulnerability publicly disclosed via a [GitHub Security Advisory](https://github.com/dronefreak/UAVidToolKit/security/advisories) once the fix is available.

---

## Scope

This project is a data-processing toolkit that reads and writes image files from a local filesystem. It does not include a web server, authentication system, or network-facing service. The most likely security concerns are:

- **Arbitrary file writes** — caused by unsanitised path inputs (e.g., path traversal via crafted filenames in a dataset).
- **Dependency vulnerabilities** — issues in `numpy`, `Pillow`, `opencv-python`, or other dependencies. Please report these upstream to the respective projects; however, if the issue specifically affects how UAVidToolKit uses a dependency, report it here.

Issues that do not have a realistic security impact (e.g., theoretical integer overflow in a research tool with no network exposure) are better handled as regular bug reports.
