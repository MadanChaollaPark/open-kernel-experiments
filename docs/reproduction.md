# Reproduction

## Offline analysis

Run from the repository root:

```bash
python3 work/verify_archive.py
python3 work/rank_run.py results/local/moe-compact-dispatch.json
```

`rank_run.py` is the final historical analyzer. Use its results only for records with compatible load/weight fields and paired timing samples. It can report a hypothetical per-case composite; that is not a separately validated implementation.

## Explicit new Modal replay

This launches billable compute. The experiment in this archive is stopped; these instructions are for a deliberate future replay.

1. Install the official `okc` CLI following the challenge site, then personally complete `okc login` and its rules acceptance.
2. Generate fresh official files with `okc init muse_moe_w4a16`. Note the directory it creates. Do not put credentials in the repository.
3. Create a local Python environment and authenticate Modal:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/modal setup
```

4. Read `source_files` in a local result to select a candidate. Example using the MoE compact-dispatch record:

```bash
.venv/bin/modal run work/replay_experiment.py \
  --record results/local/moe-compact-dispatch.json \
  --problem-dir /absolute/path/to/muse_moe_w4a16 \
  --candidate compact_all \
  --budget-usd 3 \
  --repeats 20
```

The wrapper uses the final historical remote harness, exact archived candidate source, stored benchmark cases, and extra validation cases. It loads the baseline/reference/helper files from the supplied official directory. No candidate is submitted to the challenge. Results and a separate cumulative reservation ledger go under ignored `results/replays/`.

The wrapper reserves $3 per run and refuses runs above the supplied cumulative cap. This is a conservative estimate based on the original experiment's rate assumptions, not a provider-enforced spending limit or invoice. Check current Modal pricing before use. Run it serially: the small local ledger is not a concurrent billing service. The GPU function has a 900-second timeout, no retries and at most one container. Building images or persistent storage can incur additional charges.

The image digest is pinned, but the official files you download may have changed. Compare their hashes with `result.evaluator.files` in an official receipt when assessing historical reproducibility. Replaying with the final harness also does not exactly reconstruct every earlier version of the harness.

`modal_runner_device.py` retains its original entrypoint, which requires the unpublished historical `work/challenge/` input layout and tooling ledger. Use `replay_experiment.py` for this archive. The independent Hopper probe scripts are historical implementation references and have additional cache/tooling assumptions.

## Official checks

After login, a selected source can be checked unchanged:

```bash
okc check outputs/attention_fused_hybrid.py --json
```

This is an off-board check. `okc submit` is a separate action that consumes a scored attempt; nothing in this repository runs it automatically. Official quotas, current rules and eligibility apply. Reproducing correctness or a local speedup does not guarantee the same official score.
