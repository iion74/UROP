# UROP research workspace snapshot — 2026-10-26

This release accompanies the organized public source repository. It is an
artifact export of existing work, not a new training run or a peer-reviewed
publication. See the root README for the final method and limitations.

## Assets

- `urop-policy-models.tar.gz`: 86 locally trained policy-model files, preserving
  their original `data/models/...` paths. Includes historical experiments, not
  86 variants of the final method. The final five models are
  `data/models/sequential_answer_gain_multiseed/policy_seed42..46.joblib` and are
  also available directly in the source repository.
- `urop-numeric-replay.tar.gz`: 18 numerical NPZ files under `artifacts/replay/`:
  three complete validation datasets and three training samples for each of
  five seeds. Includes IDs, numerical states, per-depth scores and cost records.
  Does not include source-document text, question text, or gold answer strings.
- `release_manifest.json` and `SHA256SUMS`: asset sizes and integrity checks.

Extract the numerical bundle at the repository root, then run:

```bash
python scripts/replay_published_policy.py --seed 42
```

Use trusted model files only: joblib/pickle/PyTorch deserialization may execute
code. See `requirements-research-snapshot.txt` for the export environment.

## Scope

The Git repository includes manuscripts, figures, review notes, historical
reports, selected small JSON reports/manifests, and final policy models.
Virtual environments, credentials, private study logs, caches, third-party
original datasets and NCS PDFs are not uploaded. Official download instructions
and a full file inventory are included. No Git LFS purchase is required.

The numerical bundle supports offline replay of the final policy, not online
latency measurement. Intermediate answer generation increases tokens and
estimated latency; the results do not demonstrate lower total computation cost.
