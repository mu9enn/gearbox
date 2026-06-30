# Final Paper Evidence Package

## Completed Structured Evidence

- Local baseline compatibility metrics were produced with `--crop-width 512`, but these are now classified as a reproducibility/provenance diagnostic artifact, not as paper-candidate evidence.
- Crop-width diagnostic caveat: old checkpoint/SHAP used 512-width input; current wavelet data are 1024-width; the near-random local result should not be used to judge collaborator historical results.
- SHAP stability table rows: 44.
- Strict feature-to-label support rows: 4.
- Weak adjacency audit rows: 44.
- Shortcut-risk candidate rows: 10.
- Underused graph-supported candidate rows: 22.

## Baseline Summary

- The crop-width local baseline numbers are not paper-candidate results and should not be used in manuscript tables.
- The paper still needs collaborator-provided baseline numeric metrics or a later formal rerun under a frozen protocol.
- Existing collaborator baseline assets remain available as figures/checkpoints under `ModelTrain/NoAttention/`, especially training curves and confusion/F1 plots.

## Main Table Candidates

- `reports/paper_evidence/final_tables/table_shap_stability_main.csv`
- `reports/paper_evidence/final_tables/table_strict_consistency_main.csv`
- `reports/paper_evidence/final_tables/table_weak_adjacency_audit_main.csv`
- `reports/paper_evidence/final_tables/table_shortcut_risk_candidates.csv`
- `reports/paper_evidence/final_tables/table_underused_causal_candidates.csv`

Diagnostic-only table:

- `reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv`

## Main Figure Candidates

- `reports/paper_evidence/final_figures/selected_main_figures.csv`
- `reports/paper_evidence/final_figures/selected_case_studies.csv`

## Strong Claims Currently Supported

- SHAP provides seed-stability rankings for channel and frequency features.
- Strict feature_to_label PC-DAG support exists for a small set of features.
- Weak label adjacency can support an attribution-graph consistency/conflict audit.

## Claims That Must Be Weakened

- Do not describe label_to_feature or undirected_with_label edges as strict causal direction.
- Do not present shortcut-risk candidates as confirmed shortcuts.
- Do not use local crop-width baseline metrics as paper evidence.

## Results Not For Strong Main Claims

- Old cross-condition results are limitation/risk only due to documented split overlap.
- GNN causal enhancement remains future work rather than a main contribution.
- CWRU/attention experiments remain optional support unless separately reproduced.
- Local crop-width baseline metrics are reproducibility diagnostics only and should not enter the evidence chain.

## Recommended First Assets For ChatGPT

- Start from `final_evidence_manifest.csv` and `final_paper_readiness_checklist.csv`.
- Do not use the local crop-width baseline paper table as Table 1. Recover collaborator numeric metrics or plan a formal rerun later.
- Use SHAP and strict/weak consistency tables for the main graph-consistency audit story.
- Use selected case studies to choose the main narrative examples.

## Remaining Minimal Points

- Author should draw the framework overview figure.
- Existing SHAP/DAG figures likely need publication styling.
- Baseline numeric evidence remains a follow-up item: recover collaborator metrics or formally rerun after protocol freeze.
