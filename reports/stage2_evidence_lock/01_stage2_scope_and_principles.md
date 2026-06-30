
# Stage 2 Scope And Principles

## Frozen Mainline

The mainline is: SHAP attribution plus dual-view PC-DAG attribution-graph consistency audit for trustworthy rotating-machinery diagnosis.

The paper should not be framed as a pure performance paper, a GNN paper, a cross-condition robustness paper, or a CWRU external-validation paper.

## Evidence Principles

- Use existing collaborator assets as the primary evidence base.
- Separate original result assets from later Codex audit/extraction tables.
- Treat SHAP/PC-DAG as a diagnostic-logic audit pipeline.
- Treat `feature_to_label` edges as strict graph support.
- Treat `label_to_feature` and undirected edges only as weak adjacency/context.
- Treat shortcut findings as candidate shortcut-risk or attribution-graph conflict, not confirmed shortcuts.
- Treat local `--crop-width 512` metrics as provenance diagnostics only.

## No-Go Actions In This Sprint

- No training.
- No checkpoint evaluation.
- No SHAP recomputation.
- No PC-KCI or GNN rerun.
- No new dataset construction.
- No CLI/config refactor.
- No manuscript prose generation.

## Status Vocabulary

- `locked_ready`: enough for Stage 3 storyboard under conservative wording.
- `provisional_needs_human_confirmation`: usable for planning, but author confirmation is required before submission.
- `blocked_needs_collaborator_confirmation`: cannot be used as final paper evidence until missing collaborator result files are recovered.
- `supplement_only`: do not use as a main claim.
- `limitation_only`: mention only as a caveat/risk.
- `future_work_only`: not part of current paper evidence.
- `do_not_use`: excluded from paper evidence.

## What Locked Means

Locked means the local evidence role and claim boundary are frozen for Stage 3 planning. It does not mean every numeric value is submission-ready. Baseline performance metrics and final PC-DAG display settings still require collaborator/human confirmation.
