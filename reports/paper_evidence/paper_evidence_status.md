# Paper Evidence Status

## Main Evidence Available

- SHAP seed-stability ranking is available: 44 rows.
- PC-DAG edge export is available: 159 edges, 31 label-related edges.
- Strict feature-to-label consistency table is available: 44 rows with counts {'low_shap_without_strict_causal_support': 32, 'low_shap_with_strict_causal_support_underused_feature': 3, 'high_shap_without_strict_causal_support_potential_shortcut': 8, 'high_shap_strict_causal_support': 1}.
- Weak adjacency consistency table is available: 44 rows with support counts {'no_label_adjacency': 18, 'strict_directed_support': 4, 'weak_label_adjacency_only': 22}.
- Baseline numeric metrics are available: 20 ok rows, crop widths ['512'], human-confirmed crop assumption=True.

## Main Evidence Still Missing

- Baseline metrics must retain the historical-width caveat: old checkpoint/SHAP used 512-width input; current data are 1024-width; use `[..., :512]` per human researcher confirmation.
- `label_to_feature` and `undirected_with_label` edges are excluded from strong causal-direction evidence.

## Supplement Candidates

- Existing confusion/F1 PNGs and attention/no-attention curves can support supplement after numeric export.
- Existing CWRU/GNN assets remain optional and should not enter the main contribution.

## Risk / Do Not Use As Strong Evidence

- Old cross-condition results remain risk/reference only due to train/cross_test exact duplicates.
- GNN causal enhancement remains future work because of the documented `classifier(x)` and scaler risks.

## Next Minimal Engineering Task

- Carry the baseline historical-width caveat into manuscript methods/reproducibility notes.
- Freeze whether manuscript uses strict causal-direction language or weaker causal-adjacency / graph-consistency language.
