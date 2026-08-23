# DataSphere — problems hit and how they were fixed (this project, 2026-08-22)

**A full guide already exists and is good**: `~/build-projs/ccm-intro/docs/compute-yandex-datasphere.md`.
Read it first — §7 has the verified `arsen4ikvar` access path, §4 has five `config.yaml` traps, §5 has
the torch/CUDA trap. This file is only the delta: what that guide does **not** cover, plus the
working invocation for this project so it doesn't have to be re-derived.

## Working setup for this project (verified 2026-08-22 23:15 MSK)

| thing | value |
|---|---|
| yc profile | `default` (this is the arsen4ikvar account; `smiles` is the dead one) |
| project | `bt12q57tmrs03pnt8drc` |
| community | `bt19h0cm8nqhr7489r9o` |
| instance | `gt4.1` (T4). `gt4i.1` also allowed. Nothing more expensive. |
| parallel cap | **4 jobs total**, self-imposed by the user's instruction |

```bash
export PATH="$HOME/.local/bin:$HOME/yandex-cloud/bin:$PATH"
GRPC_DNS_RESOLVER=native datasphere --profile default project job execute \
  -p bt12q57tmrs03pnt8drc -c config.yaml
GRPC_DNS_RESOLVER=native datasphere --profile default project job list -p bt12q57tmrs03pnt8drc
datasphere project job cancel --id <job-id>          # no -p needed
datasphere project job download-files --id <job-id> --with-logs --output-dir ./out
```

## New problems, not in the guide

**1. `requirements-file` cannot contain a bare `--index-url` line.**
The guide's own recommended fix for the CUDA-13 trap is
`pip install torch --index-url https://download.pytorch.org/whl/cu121`, put in a requirements file.
That crashes submission locally:
```
packaging.requirements.InvalidRequirement: Expected package name at the start of dependency specifier
    --index-url https://download.pytorch.org/whl/cu121
```
The CLI parses each line with `packaging.Requirement`, which rejects pip flags.
**Fix:** keep `requirements.txt` to plain packages (`numpy`) and put the pinned install in `cmd:`
```yaml
cmd: pip install -q torch --index-url https://download.pytorch.org/whl/cu121 && python main.py
```
This keeps the cu121 guarantee (avoiding the silent `torch.cuda.is_available()==False` on a T4)
without going through the parser.

**2. Wrapping `job execute` in `timeout` (or any harness that backgrounds it) silently double-submits.**
The guide warns the poller can die while the job survives. The same thing happens if *you* kill the
poller: the job was already created server-side. Under an agent harness that moves a long
foreground command to the background, a "retry" creates a **second paying job**. Hit this exactly:
two `tlab-exit-dump` jobs `EXECUTING` four minutes apart.
**Fix / rule:** `project job list` **before every resubmit**, without exception, and cancel the
duplicate immediately (`project job cancel --id <id>` — takes effect in ~1s, status goes
`CANCELLING`). Treat submission as non-idempotent.

**3. `inputs:` vs `local-paths:` — large binaries go in `inputs:`.**
Not a failure, a working pattern worth recording: code files (`main.py`, imported modules) go in
`env.python.local-paths`; data/weights (a 35MB `.pt`, a 1.7MB `.npz`) go in a top-level `inputs:`
list. Both arrive in the job's working directory.

## Working config.yaml for this project

```yaml
name: tlab-exit-dump
desc: ...
cmd: pip install -q torch --index-url https://download.pytorch.org/whl/cu121 && python main.py
env:
  python:
    type: manual          # NOT auto
    version: "3.11"       # 3.8-3.12 only; NOT your local 3.14
    requirements-file: requirements.txt
    local-paths: [main.py, _dump.py, model.py]
inputs: [ckpt.pt, frozen_eval_set.npz]
outputs: [results/**]     # WITHOUT THIS NOTHING SURVIVES THE JOB
cloud-instance-type: gt4.1
```
Plus the guide's trap that still bites: the entry script **must** contain a literal
`if __name__ == "__main__":`, and keep heavy imports (`torch`) *inside* that guard — the CLI
`exec()`s the entry script locally during dependency discovery, and a top-level `import torch` fails
there because the CLI's own pipx venv has no torch.

## Always
Print `torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)` as the job's
first action and **exit non-zero if CUDA is missing** — a GPU job that quietly runs on CPU bills in
full and succeeds at everything except its purpose.

---

## Measured cost of a job, end to end (timing probe `bt1tci30t1rif3ui0bi9`, 2026-08-22)

Reconstructed from the job record (`created_at` → `finished_at`) against an epoch printed by the
script as its first action.

| phase | sec | % of e2e |
|---|---|---|
| **SETUP** — container start, venv create, pip (numpy + torch cu121), local modules, inputs | **218.2** | **83.9%** |
| python side total | 14.6 | 5.6% |
| · `import torch` | 6.0 | 2.3% |
| · actual GPU compute | 7.2 | 2.8% |
| · torch.load 35MB ckpt + build model + `.to(cuda)` + np.load | 0.18 | 0.1% |
| **TEARDOWN** — output collection + container destroy | **27.2** | **10.5%** |
| end-to-end | 260.0 | |

**Fixed overhead ≈ 245s ≈ 4.1 min per job, independent of compute length.** So: 21% of a 15-min job,
6.4% of a 1h job, 1.7% of a 4h job, 0.7% of a 10h job. **Use DataSphere for multi-hour jobs; batch
small work into one job rather than submitting several.** (This is why the exit dump and the
rule-fitting were merged into a single job instead of chained.)

Loading a 35MB checkpoint costs 0.021s — input size is not the problem. Inputs are already on disk
when the script starts (`stat` = 0.000s), so upload happens during setup, before `created_at`
elapses on the client side.

**Measurement gap, with the fix:** `system.log` has 23 `[INFO]` lines and **no timestamps**, so the
218s cannot be split into container / venv / pip-torch / inputs from the logs alone. Fix in one line
— make `cmd` stamp the clock around the install:
```yaml
cmd: date +%s; pip install -q torch --index-url https://download.pytorch.org/whl/cu121; date +%s; python main.py
```

## Killing the 218s: use a public Docker image with torch preinstalled

**Notebooks ≠ Jobs.** The Yandex article "Установить зависимости" (preinstalled DS/ML packages,
`%pip install`) describes **JupyterLab notebooks in a project**. DataSphere **Jobs** build a fresh
isolated venv per job — my own log shows it downloading **numpy** from PyPI, which is about as
"preinstalled" as a package gets. Do not assume the notebook environment's packages exist in a job.

**The cheap fix is NOT build-and-push.** `ccm-intro` §6 already documents the pattern and it is an
inline public image tag — no Dockerfile, no registry upload:
```yaml
env:
  docker:
    image: nvidia/cudagl:11.4.2-runtime-ubuntu20.04
```
That section also records `torch 2.4.1+cu121` working inside such an image, and notes §5's
3.10-venv caveat applies to the *default* image, not to custom ones. So pointing at e.g.
`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` should remove the torch install entirely. Yandex
caches the image after the first pull (the stock image already logs `Image '...' exists`), so the
multi-GB pull is paid once, not per job.

I earlier priced a custom image at 1-2h on the assumption it meant building and pushing my own —
**that was wrong**, and it is the reason to prefer this over the conditional below when N is more
than a couple of jobs.

**Belt-and-braces predicate**, useful either way and self-verifying — it uses the *stock* image's
torch only if that torch's CUDA actually works on the T4, so it cannot be fooled by the
CUDA-13-wheel-on-a-T4 trap (§5) that makes `cuda.is_available()` False while the job "succeeds":
```yaml
cmd: python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
     || pip install -q torch --index-url https://download.pytorch.org/whl/cu121; python main.py
```

**When a custom image is the right call:** many jobs, or build-from-source deps. Under a deadline
with N≈3 jobs the arithmetic is different — 218s × 3 ≈ 11 min saved. Judge by N, not by principle.

## Early-stop safety — TESTED, and it is the fact that unlocks long jobs

| job state when you pull | stdout recovered? | declared `outputs:` recovered? |
|---|---|---|
| **EXECUTING** | **no** | **no** — `"job is still running (2), nothing to download yet"` |
| cancelled during SETUP | no (system.log only) | no |
| **cancelled MID-COMPUTE** | **yes** | **yes** |
| ERROR after compute finished | yes | no |
| SUCCESS | yes | yes |

**The operational consequence, and it is the important one:** you cannot peek at a running job's
**files**. To harvest a long job's *outputs* early you must **cancel it** — cancelling is the harvest
mechanism, not a loss.

> **CORRECTED 2026-08-23 14:00 — "you cannot peek at a running job" was too strong, and I had been
> repeating it as if it covered everything.** `attach` works, and it is the difference between
> watching a 6-hour job and guessing at it. Tested live on `tlab-deep-full` mid-run:
>
> ```
> timeout 75 env GRPC_DNS_RESOLVER=native datasphere project job attach --id <job-id>
> 2026-08-23 13:41:05 - [11292.5s]  step 7100/19531 loss=4.3160 n_loops=46 gnorm=0.50 1296 tok/s
> ```
>
> **One line was enough** to get progress (36%), throughput (1296 tok/s), loss, and an ETA — with no
> cancellation and no cost. What you genuinely cannot get is **history from before you attached**,
> and **files** while it runs. Practical notes: wrap it in `timeout` (attach blocks indefinitely),
> expect to re-attach (its poller dies on the documented `auth.py` AssertionError), and grep for your
> job's own log prefix because gRPC fork-handler noise interleaves with real output.

Tested with a purpose-built job (`ds_killtest`) that heartbeats to stdout and rewrites a results
file every 3s, cancelled mid-compute: `download-files --with-logs` returned `stdout.log` **and**
`partial.json`. The CLI still warns *"was completed with error (6). Not all files can be
downloaded"* — that warning does NOT mean nothing survived; check what actually landed.

**Consequence: a DataSphere job does not have to fit your deadline.** Launch it, harvest it when you
need to, cancel the rest. Two requirements to actually get this: (1) write results **incrementally**
(our kernels dump `results.json` + a checkpoint at every eval boundary, not at the end), and
(2) order arms **cheapest-first** so an early stop loses the least.

**stdout lags.** A job cancelled after ~4 min of compute returned only ~36s of heartbeats. Expect to
lose the last minutes of printed output; the incrementally-written file is the more reliable channel.

**Torch install ≈ 2.5-3 min of the 218s setup.** Inferred by contrast: the killtest (numpy only, no
torch install) reached compute in ~1 min, against ~3.6 min for a torch-installing job.

## `pip`'s root-user warning marks the whole job ERROR — and that costs you `outputs:`

A job whose compute finished perfectly was reported **ERROR**, and its declared `outputs:` were not
collected. The only `[ERROR]` line in `system.log` was pip's advisory:

> `WARNING: Running pip as the 'root' user can result in broken permissions... Use the
> --root-user-action option if you know what you are doing and want to suppress this warning.`

Anything pip writes to stderr appears to be promoted to an error for the job's status, and an ERROR
job does not hand back files (stdout still survives — which is the second time defensive printing
saved a result here). **Fix: `pip install -q --root-user-action=ignore ...` in every `cmd`.** Applied
to all job configs in this repo.

## The docker fix WORKS — measured

`env.docker.image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` plus the conditional predicate
produced, in the job's own stdout:
```
1787432007
TORCH_PREINSTALLED_OK skipping pip
1787432009
[    0.0s] CUDA available: True, device: Tesla T4
```
**2 seconds instead of ~150s of pip install.** The image's torch passes the `cuda.is_available()`
gate, so the pinned install never runs. Use this for every job.

## wandb: the mechanism works, but 25 of 26 job dirs never used it

Audited 2026-08-23 after I asserted — wrongly, and the user pushed back — that "every DS job ran with
zero live logging". One did.

| | count |
|---|---|
| `ds_*` dirs with wandb in `requirements.txt` **and** imported in `main.py` | **1** (`ds_probe`) |
| dirs carrying `export WANDB_API_KEY="$WANDB_API_KEY"` in `cmd` and nothing else | 15 |
| dirs with neither | 10 |
| `kaggle/main.py` | none |

**The `export` line in those 15 configs is a no-op.** `cmd` runs on the *remote* node, where
`$WANDB_API_KEY` is unset, so it expands to empty — and nothing in those `main.py` files imports
wandb, so it would not have mattered anyway. It was copied from `ds_probe` without the two parts that
make it work.

**`ds_probe` shows the mechanism that does work**: a top-level `WANDB_API_KEY:` mapping in
`config.yaml`, which DataSphere passes into the job environment. Its docstring says why it exists —
*"the local `datasphere job attach` stream died 13 times today... metrics that land server-side in
wandb do not depend on a local process staying alive at all."* That reasoning is still right even
now that `attach` is known to work, because attach needs a live local process and wandb does not.

**And that mapping is exactly how the API key leaked.** It held the real key inline in a tracked
file — the 18-config / 18-commit leak in §6.0. It now reads `REDACTED_SET_VIA_ENV`.
**Rule: generate any job dir that carries a key into the scratchpad and launch from there. Never
into the repo.**
