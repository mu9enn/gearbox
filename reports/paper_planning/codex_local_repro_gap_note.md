# Codex Local Reproducibility Gap Note

The crop-width-512 evaluation is treated as a local reproducibility/provenance diagnostic artifact, not as a paper candidate result.

## What Was Done Locally

- Codex previously evaluated existing NoAttention checkpoints against the current local `wavelet_dataset/*.npz`.
- Because the saved checkpoints expect width 512 while the current local wavelet arrays have width 1024, the evaluation used `[..., :512]`.
- The run produced structured files under `reports/paper_evidence/baseline_metrics/`.

## Why The Result Is Near Random

- The recovered local metrics are close to random-classification behavior.
- This likely reflects a provenance mismatch among current local code, current local data files, and saved checkpoints.
- The current repository contains 1024-width wavelet files, while legacy checkpoint/SHAP assets indicate a 512-width historical input protocol.
- The exact collaborator data contract and original metric export path are not fully present in the current local project state.

## Why This Cannot Replace Collaborator Historical Results

- The local crop-width evaluation is not the collaborator's original reported experiment.
- It was executed as a compatibility diagnostic after discovering checkpoint/data width mismatch.
- It should not be used to conclude that the collaborator's historical baseline was random.
- It should not be used as a paper Table 1 result or as evidence against the topic.

## What It Shows

- It shows a reproducibility/provenance gap in the current local assets.
- It shows that the current repository needs clearer dataset contracts, checkpoint metadata, metric logs, config snapshots, and provenance notes before formal reproduction.
- It does not invalidate the SHAP/PC-DAG paper topic or the collaborator's already generated result figures/caches.

## Current-Stage Decision

- Do not continue chasing `--crop-width 512` locally in the paper-planning phase.
- Do not run more checkpoint evals now.
- Do not use the crop-width metrics as paper evidence.
- Treat the crop-width result only as a local diagnostic artifact and planning warning.

## Future Formal Reproducibility Stage

If the project later enters a formal reproduction sprint:

- Recover the collaborator's original baseline numeric metrics, logs, or result tables first.
- If unavailable, freeze a new clean evaluation protocol before rerunning.
- Record dataset version, split contract, input width, normalization source, class mapping, checkpoint hash, code commit, random seeds, and metric export script.
- Export condition-level mean/std and classification reports from the formal protocol.
- Keep cross-condition evidence separate unless a clean leakage-free split is explicitly created and approved.
