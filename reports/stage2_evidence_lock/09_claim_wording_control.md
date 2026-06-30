
# Claim Wording Control

## Can Write Strongly

- The project implements an attribution-graph audit workflow combining SHAP attribution and dual-view PC-DAG analysis.
- SHAP provides channel-level and wavelet-frequency-level attribution patterns for the trained diagnostic model.
- Existing PC-DAG assets provide channel-view and frequency-view graph structures for auditing diagnostic logic.
- The strict audit table uses only `feature_to_label` edges as directional graph support.
- The weak audit table separately records label adjacency from `label_to_feature` or undirected edges.
- The method can identify attribution-graph consistency, conflict, and candidate shortcut-risk patterns.

## Must Weaken

- Use "graph support", "diagnostic-logic audit", "attribution-graph consistency", or "candidate shortcut-risk" instead of confirmed causal mechanism.
- Use "PC-DAG-derived dependency structure" instead of ground-truth causal graph.
- Use "may indicate", "suggests", "is consistent with", and "flags for review" for shortcut/conflict cases.
- Use "cross-condition result is exploratory / limited by split overlap risk" if cross results are mentioned.
- Use "CWRU/GNN branch is exploratory or future work" unless confirmed metrics and implementation fixes are supplied.

## Cannot Write

- Do not claim the model discovers true physical causality.
- Do not claim `label_to_feature` edges prove causal influence from features to label.
- Do not claim undirected PC-DAG links are directional causal evidence.
- Do not claim shortcut features are confirmed shortcuts without intervention or leak-free robustness tests.
- Do not claim cross-condition robustness from the current cross split.
- Do not cite local `--crop-width 512` metrics as paper performance evidence.
- Do not frame GNN as a validated main contribution in the current paper.
- Do not present CWRU as a completed external validation unless collaborator numeric metrics are recovered.

## Recommended Main Claim Template

We propose a diagnostic-logic audit framework that couples SHAP attribution with dual-view PC-DAG structures to examine whether model-relevant channels and wavelet bands are consistent with label-related graph dependencies. The framework distinguishes strict feature-to-label graph support from weaker label adjacency, enabling conservative identification of attribution-graph agreement, conflict, and candidate shortcut-risk patterns.
