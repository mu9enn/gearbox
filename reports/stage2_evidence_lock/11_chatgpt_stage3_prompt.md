
# Prompt For ChatGPT Stage 3

You are ChatGPT leading the scientific design of the gearbox project. Use the Stage 2 evidence-lock package as the local evidence boundary.

## Frozen Direction

Frame the paper around SHAP attribution plus dual-view PC-DAG attribution-graph consistency audit for trustworthy rotating-machinery diagnosis.

Do not frame the paper as:

- a pure model-performance paper;
- a GNN paper;
- a cross-condition robustness paper;
- a CWRU external-validation paper.

## Locked Evidence

- SHAP attribution assets and multi-seed summaries exist.
- Dual-view PC-DAG assets and exported edge summaries exist.
- Strict/weak consistency audit tables exist.
- Case-study candidates exist.
- Conservative claim wording is defined in `09_claim_wording_control.md`.

## Blocked Evidence

- Trusted original baseline numeric metrics are missing.
- PC-DAG final setting needs confirmation.
- SHAP cache/config provenance needs confirmation.
- Local `--crop-width 512` metrics are diagnostic-only and must not be used as paper evidence.

## Required Stage 3 Outputs

1. A contribution map with 2-3 defensible contributions.
2. A main-figure storyboard, including pipeline, SHAP, dual-view PC-DAG, and case-study panels.
3. A final table plan separating locked, provisional, and blocked tables.
4. A claim hierarchy using conservative causal language.
5. A minimal follow-up/confirmation plan for the collaborator.
6. A journal-positioning assessment that avoids overclaiming causality or robustness.
7. A supplement/future-work policy for attention, GNN, CWRU, and cross-condition.

## Wording Constraints

- Use "diagnostic-logic audit" and "attribution-graph consistency".
- Only `feature_to_label` edges can be called strict graph support.
- Treat `label_to_feature` and undirected edges as weak adjacency.
- Treat shortcut findings as candidate shortcut-risk, not confirmed shortcut mechanisms.
- Do not use local crop-width-512 metrics.
