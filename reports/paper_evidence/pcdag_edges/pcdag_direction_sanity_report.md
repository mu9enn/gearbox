# PC-DAG Direction Sanity Report

## Facts Checked

- `Common/PCCausal.py` and `tools/export_pcdag_edges.py` both build background knowledge intended to forbid `label -> feature`.
- The PC input column order is mapped back from causallearn's `X1...Xn` nodes to feature names plus `label`.
- Edge endpoints are interpreted conservatively: `TAIL/ARROW` means source -> target; `ARROW/TAIL` means target -> source; other endpoint combinations are treated as undirected/partially oriented.
- The exported edge table is based on the original PC graph, not the later visualization-only filtered graph.

## Direction Finding

- `feature_to_label`: 4.
- `label_to_feature`: 22.
- `undirected_with_label`: 5.

## Interpretation

- The most likely cause of `label_to_feature` rows is that the original causallearn PC graph still contains orientations that should not be used as physical causal direction evidence.
- The visualization code's filtered graph keeps only feature -> label edges for the displayed filtered DAG, but that filtered graph was not previously persisted as an edge CSV.
- Endpoint reversal is possible in any custom exporter, but the observed `TAIL/ARROW` string representation matches causallearn's printed `X --> Y` convention in smoke tests.
- Therefore `feature_to_label` can be used as strong evidence; `undirected_with_label` can be weak adjacency; `label_to_feature` should be discarded from strong evidence.
