
# Stage 2 Evidence Lock Package

This folder freezes local evidence sources before ChatGPT Stage 3 deep research and storyboard design. It is an evidence-control package, not manuscript prose.

## Reading Order

1. `01_stage2_scope_and_principles.md`
2. `08_evidence_lock_decision_register.csv`
3. `03_baseline_numeric_metrics_status.csv`
4. `04_checkpoint_shap_pcdag_provenance_map.csv`
5. `05_final_table_figure_source_map.csv`
6. `06_pcdag_setting_lock_candidates.csv`
7. `07_case_study_lock_candidates.csv`
8. `09_claim_wording_control.md`
9. `10_submission_readiness_after_stage2.md`
10. `02_author_result_request_packet.md`
11. `11_chatgpt_stage3_prompt.md`

## Current Evidence Snapshot

- Strict consistency counts: high_shap_strict_causal_support=1; high_shap_without_strict_causal_support_potential_shortcut=8; low_shap_with_strict_causal_support_underused_feature=3; low_shap_without_strict_causal_support=32
- Weak adjacency counts: no_label_adjacency=18; strict_directed_support=4; weak_label_adjacency_only=22
- PC-DAG label-relation counts: not_available

## Non-Goals

This package did not train, evaluate checkpoints, recompute SHAP, rerun PC-KCI/GNN, build new datasets, or revise algorithms. The local `--crop-width 512` result is diagnostic-only and is not paper evidence.
