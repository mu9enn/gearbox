# Final Paper Evidence Package

## Completed Structured Evidence

- Baseline metrics were recovered for existing NoAttention checkpoints on `test_set` with `--crop-width 512`.
- Historical-width caveat: old checkpoint/SHAP used 512-width input; current wavelet data are 1024-width; human researcher confirmed using first 512 points via [..., :512] to recover historical evaluation protocol.
- SHAP stability table rows: 44.
- Strict feature-to-label support rows: 4.
- Weak adjacency audit rows: 44.
- Shortcut-risk candidate rows: 10.
- Underused graph-supported candidate rows: 22.

## Baseline Summary

- Bearing_20_0: accuracy 0.2168 +/- 0.0168; macro-F1 0.1097 +/- 0.0367; weighted-F1 0.1097 +/- 0.0367; n=680; seeds=5.
- Bearing_30_2: accuracy 0.2037 +/- 0.0075; macro-F1 0.0801 +/- 0.0267; weighted-F1 0.0801 +/- 0.0267; n=1020; seeds=5.
- Gear_20_0: accuracy 0.1997 +/- 0.0006; macro-F1 0.0667 +/- 0.0000; weighted-F1 0.0667 +/- 0.0000; n=680; seeds=5.
- Gear_30_2: accuracy 0.1959 +/- 0.0105; macro-F1 0.0949 +/- 0.0180; weighted-F1 0.0949 +/- 0.0180; n=1020; seeds=5.

## Main Table Candidates

- `reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv`
- `reports/paper_evidence/final_tables/table_shap_stability_main.csv`
- `reports/paper_evidence/final_tables/table_strict_consistency_main.csv`
- `reports/paper_evidence/final_tables/table_weak_adjacency_audit_main.csv`
- `reports/paper_evidence/final_tables/table_shortcut_risk_candidates.csv`
- `reports/paper_evidence/final_tables/table_underused_causal_candidates.csv`

## Main Figure Candidates

- `reports/paper_evidence/final_figures/selected_main_figures.csv`
- `reports/paper_evidence/final_figures/selected_case_studies.csv`

## Strong Claims Currently Supported

- Existing NoAttention CNN checkpoints can be evaluated on the historical 512-width input protocol, with the caveat recorded.
- SHAP provides seed-stability rankings for channel and frequency features.
- Strict feature_to_label PC-DAG support exists for a small set of features.
- Weak label adjacency can support an attribution-graph consistency/conflict audit.

## Claims That Must Be Weakened

- Do not describe label_to_feature or undirected_with_label edges as strict causal direction.
- Do not present shortcut-risk candidates as confirmed shortcuts.
- Do not describe baseline metrics without the 512-width historical data caveat.

## Results Not For Strong Main Claims

- Old cross-condition results are limitation/risk only due to documented split overlap.
- GNN causal enhancement remains future work rather than a main contribution.
- CWRU/attention experiments remain optional support unless separately reproduced.

## Recommended First Assets For ChatGPT

- Start from `final_evidence_manifest.csv` and `final_paper_readiness_checklist.csv`.
- Use baseline paper table for Table 1 with caveat.
- Use SHAP and strict/weak consistency tables for the main graph-consistency audit story.
- Use selected case studies to choose the main narrative examples.

## Remaining Minimal Points

- Author should draw the framework overview figure.
- Existing SHAP/DAG figures likely need publication styling.
- Manuscript methods must state the 512-width historical input caveat.
