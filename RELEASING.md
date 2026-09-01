# Releasing

Publishing is automated: **creating a GitHub Release publishes that version to
PyPI.** Everything below the one-time setup is a three-command routine.

Authentication uses **Trusted Publishing** — GitHub proves the workflow's
identity to PyPI over OIDC, and PyPI checks it against a publisher you
registered. No API token is created, stored, or able to leak. This also means
nothing can publish `pymodest` except this repository's `publish.yml`.

---

## One-time setup

### 1. Push the repository

```bash
gh repo create BioJazz-Lab/pyModEst --public --source=. --remote=origin --push
```

Trusted Publishing identifies the workflow by owner, repository and file name,
so the repo must exist first.

### 2. Register the pending publisher on PyPI

The project does not exist on PyPI yet, so register a **pending** publisher —
it becomes a normal one on first upload.

1. Sign in at [pypi.org](https://pypi.org) (2FA is required on all accounts).
2. Go to **Your projects → Publishing**, or
   <https://pypi.org/manage/account/publishing/>.
3. Under *Add a new pending publisher*, choose GitHub and fill in:

   | field | value |
   | --- | --- |
   | PyPI project name | `pymodest` |
   | Owner | `BioJazz-Lab` |
   | Repository name | `pyModEst` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

The environment name must match, and it is what makes this safe: it lets you
require a manual approval before any upload.

### 3. Do the same on TestPyPI

Repeat at <https://test.pypi.org/manage/account/publishing/> with environment
name `testpypi`. TestPyPI is a separate site with its own account.

### 4. Create the GitHub environments

In the repo: **Settings → Environments → New environment**, named `pypi` and
`testpypi`. For `pypi`, add yourself under *Required reviewers* — then every
release waits for one click before anything reaches PyPI. This is the cheapest
possible guard against an accidental release, and worth having because **a
version published to PyPI can never be replaced or reused.**

---

## Rehearse on TestPyPI

Do this once before the first real release.

**Actions → publish → Run workflow → target: `testpypi`**

Then check the upload installs and runs:

```bash
uv run --with pymodest --index https://test.pypi.org/simple/ \
       --index-strategy unsafe-best-match --no-project pymodest --version
```

(`--index-strategy unsafe-best-match` lets the dependencies come from real
PyPI while `pymodest` comes from TestPyPI.)

---

## Releasing a version

### 1. Bump the version

```bash
uv version 0.2.0        # edits pyproject.toml and refreshes uv.lock
```

Or edit `version` in `pyproject.toml` by hand and run `uv lock`.

### 2. Commit and push

```bash
git commit -am "Release 0.2.0"
git push
```

Wait for the `tests` workflow to pass. It runs the suite on Linux, macOS and
Windows across Python 3.11–3.13, re-runs the documented examples, and builds
the distributions.

### 3. Create the release

```bash
gh release create v0.2.0 --generate-notes
```

Or use the GitHub UI. The tag must be the version with a `v` prefix — the
workflow refuses to publish if `v0.2.0` does not match `version = "0.2.0"` in
`pyproject.toml`, which is the single most common way a release goes wrong.

The `publish` workflow then runs the tests again, builds, waits for your
approval on the `pypi` environment, and uploads.

---

## Version numbers

`pymodest` is pre-1.0, so the compatibility promise is weak by convention:

- **0.x.y → 0.x.(y+1)** — fixes, documentation, internal changes.
- **0.x.y → 0.(x+1).0** — new capability, or a breaking change to the TOML
  schema or the Python API.

Reserve 1.0.0 for when the configuration schema is stable enough that you are
willing to keep it working.

---

## If something goes wrong

**A bad version reached PyPI.** You cannot fix it in place. Yank it
(*Manage → Options → Yank*), which hides it from new installs while leaving
existing pins working, then publish the next patch version. Deleting a release
does **not** free the version number for reuse.

**The workflow fails with an OIDC or "not authorized" error.** The publisher's
owner, repository, workflow filename and environment must match exactly,
including case. Check the pending publisher on PyPI against
`.github/workflows/publish.yml`.

**`uv sync --locked` fails in CI.** `uv.lock` is out of date with
`pyproject.toml`. Run `uv lock` and commit the result.

**A platform fails in CI but not locally.** `antimony` and `libroadrunner`
resolve to different versions per platform; see the table in
[docs/troubleshooting.md](docs/troubleshooting.md).
