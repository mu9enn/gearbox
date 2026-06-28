# Consistency Notes

- SHAP source: `reports/paper_evidence/shap_seed_stability_summary.csv`.
- Edge source: `reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv`.
- High-SHAP threshold: mean rank <= 2.
- Rows: 44; summaries: 8.
- Consistency counts: {'low_shap_no_label_edge_uninformative': 16, 'low_shap_label_edge_underused_causal_feature': 3, 'unknown_edge_direction': 22, 'high_shap_label_edge_consistent': 1, 'high_shap_no_label_edge_potential_shortcut': 2}.
- `feature_to_label` edges are treated as attribution-causality consistency when the feature is high-SHAP.
- Features with high SHAP but no label-related PC-DAG edge are marked as potential shortcut candidates.
