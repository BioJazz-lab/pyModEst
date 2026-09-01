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

Two environments are needed, `pypi` and `testpypi`, matching the `environment:`
blocks in `publish.yml` and the *Environment name* fields registered on PyPI.

**What the environment is doing here.** It holds no secrets — Trusted
Publishing has none. It does two other jobs:

1. **It narrows the OIDC identity.** When a job declares `environment: pypi`,
   GitHub adds an `environment` claim to the token it mints. PyPI checks that
   claim against the publisher you registered, so a workflow that runs
   *without* the environment cannot publish even if it is otherwise identical.
2. **It is the gate.** Required reviewers pause the run until a human clicks
   approve.

#### Creating them

**Settings → Environments → New environment** → name it → *Configure
environment*. Names are case-insensitive and must be unique in the repo.

Or through the API:

```bash
gh api --method PUT repos/BioJazz-Lab/pyModEst/environments/testpypi

USER_ID=$(gh api user --jq .id)
gh api --method PUT repos/BioJazz-Lab/pyModEst/environments/pypi --input - <<JSON
{"reviewers": [{"type": "User", "id": $USER_ID}]}
JSON
```

#### What to configure

| environment | setting | value |
| --- | --- | --- |
| `pypi` | Required reviewers | yourself (up to 6 people or teams; one approval releases the job) |
| `pypi` | Prevent self-review | **off** — a solo maintainer who enables this can never approve their own release |
| `pypi` | Deployment tag rule | `v*`, so only release tags can deploy to this environment |
| `testpypi` | — | nothing; it exists only to scope the OIDC claim |

#### The plan restriction

This is the part that decides whether the gate is available at all:

| repository | environments | required reviewers / wait timer |
| --- | --- | --- |
| **public** | all plans | **all plans** |
| private or internal | Pro, Team or Enterprise | Enterprise only |

On a **private** repo on a Free, Pro or Team plan you can create the
environment — enough for the OIDC claim — but **not** add required reviewers.
If you want the approval gate without an Enterprise plan, the repository has to
be public.

#### A trap worth knowing

If `publish.yml` names an environment that does not exist, GitHub **creates it
silently** on first run, with no protection rules. The workflow succeeds and
publishes, and nothing warns you that the gate you thought you configured was
never there. Create the environments before the first release, and confirm they
are listed under Settings → Environments with the reviewer shown.

#### What a gated release looks like

After you publish a GitHub Release, the `publish` workflow builds and tests,
then the `pypi` job stops with **Waiting for review**. Reviewers get a
notification; open the run, click **Review deployments**, tick `pypi`, and
**Approve and deploy**. Only then does anything reach PyPI. Rejecting it leaves
the release in place with nothing published — you can fix and re-run.

---

## Rehearse on TestPyPI

Do this once before the first real release.

**Actions → publish → Run workflow → target: `testpypi`**

Then check the upload installs and runs. **Take the dependencies from real
PyPI and only `pymodest` from TestPyPI** — TestPyPI does not mirror PyPI, and
it carries stale test uploads of common names like `pandas`:

```bash
uv venv /tmp/pymodest-check
uv pip install --python /tmp/pymodest-check numpy scipy pandas antimony libroadrunner
uv pip install --python /tmp/pymodest-check --no-deps \
    --index https://test.pypi.org/simple/ pymodest
/tmp/pymodest-check/bin/pymodest --version
```

Two steps, but no cross-index resolution can go wrong: the dependencies come
from PyPI, then `--no-deps` installs your artifact alone.

The one-liner version works when TestPyPI happens not to shadow a dependency:

```bash
uv run --with pymodest --index https://test.pypi.org/simple/ \
       --index-strategy unsafe-best-match --no-project pymodest --version
```

Without `--index-strategy unsafe-best-match` this fails with
`no versions of pandas`: `--index` gives TestPyPI priority, and uv's default
`first-index` strategy stops at the first index where a name exists rather than
falling through to PyPI. Note that `unsafe-best-match` is named for a real
risk — it lets any configured index supply any package — so keep it to one-off
checks, never a project's default.

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
