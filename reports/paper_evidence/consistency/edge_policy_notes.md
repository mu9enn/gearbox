# Edge Policy Notes

- Exported PC-DAG edges: 159.
- Label-related edges: 31.
- Edge relation counts: {'not_label_related': 128, 'feature_to_label': 4, 'label_to_feature': 22, 'undirected_with_label': 5}.
- Strong `feature_to_label` edges retained for strict causal-support evidence: 4.
- `label_to_feature` edges discarded from strong causal support: 22.
- `undirected_with_label` edges retained only as weak adjacency: 5.

## Policy

- Strict table: only `feature_to_label` supports directional attribution-causality consistency.
- Adjacency table: any label-related edge can support a weaker graph-adjacency statement, without directional causal language.
- `label_to_feature` is not used as causal support because label is a diagnosis/fault class and should not be interpreted as causing vibration features.

## Recommendation

- Strong wording should be limited to strict `feature_to_label` cases.
- If strict cases are too sparse for a broad claim, use `attribution-causal-adjacency audit` or `graph-consistency audit` language.
