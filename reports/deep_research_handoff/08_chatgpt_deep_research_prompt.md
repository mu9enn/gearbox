
# ChatGPT Deep Research Prompt

You are ChatGPT taking over the deep research, literature positioning, contribution design, and SCI Q1 paper planning for the GearBox project. Use the local facts prepared by Codex, but do not treat Codex's diagnostic scripts as the original scientific contribution.

## Current Project Facts

- Topic area: rotating machinery / gearbox / bearing fault diagnosis.
- Main available pipeline: raw multichannel vibration -> angle-domain and wavelet tensor representation -> CNN/no-attention diagnosis -> SHAP channel/frequency attribution -> SHAP-weighted PC_Datasets -> channel-view and frequency-view PC-DAG -> attribution-graph consistency/conflict audit.
- Key local evidence directories: `ModelTrain/`, `CalculateShapValues/`, `PC_Datasets/`, `DAG/PC_DAG/`, `GNNCausal/`, `CWRU/`, `DAG/PureData`, `DAG/Noise`.
- Key handoff files: `reports/deep_research_handoff/*`, `reports/paper_planning/*`, `reports/paper_evidence/*`.

## Collaborator Core Contribution To Respect

The collaborator's contribution appears to be a diagnostic-logic audit framework, not merely a classifier:

1. Build wavelet multichannel representations for rotating-machinery diagnosis.
2. Train/use CNN diagnosis models as the target of explanation.
3. Compute SHAP attribution over channels and wavelet frequency bands.
4. Convert SHAP-weighted features into dual-view PC-DAG inputs.
5. Use channel-view and frequency-view graph structures to audit attribution logic.
6. Identify consistency, conflict, shortcut-risk candidates, and underused graph-supported features.

Do not rewrite this as a GNN paper unless the human researcher explicitly asks.

## Existing Evidence

- SHAP stability: 44 rows in `shap_seed_stability_summary.csv`; 220 per-seed SHAP rank rows.
- PC-DAG edge export: 159 total edges and 31 label-related edges.
- Strict feature-to-label support counts: {'low_shap_without_strict_causal_support': 32, 'low_shap_with_strict_causal_support_underused_feature': 3, 'high_shap_without_strict_causal_support_potential_shortcut': 8, 'high_shap_strict_causal_support': 1}.
- Weak adjacency support counts: {'no_label_adjacency': 18, 'strict_directed_support': 4, 'weak_label_adjacency_only': 22}.
- Shortcut-risk candidates: 10 rows.
- Underused graph-supported candidates: 22 rows.
- Existing figures: SHAP total graphs, PC-DAG figures, training curves, confusion/F1 figures, attention comparison figures, GNN/CWRU curves.

## Evidence Gaps

- Trusted collaborator baseline numeric metrics are not currently indexed.
- Local `--crop-width 512` baseline evaluation is diagnostic only and near-random; do not treat it as collaborator result failure.
- PC-DAG direction must be conservative: `feature_to_label` is strict support; `label_to_feature` and undirected label-related edges are weak adjacency only.
- Cross-condition evidence has split-overlap risk and should not support strong robustness claims.
- GNN has implementation/provenance risks and should remain future work unless repaired and verified.
- CWRU and attention assets are currently support/future unless numeric tables are recovered.

## Recommended Storyline

Prioritize Storyline A:

Attribution-graph consistency audit for trustworthy rotating machinery diagnosis.

Possible phrasing direction: an explainable diagnostic logic audit framework that combines SHAP attribution with dual-view causal/graph discovery to identify attribution-graph agreement, conflict, shortcut-risk candidates, and underused graph-supported features.

## Literature Directions To Research

1. Explainable AI for rotating machinery fault diagnosis: SHAP, Grad-CAM, integrated gradients, attention explanations.
2. Shortcut learning and spurious correlation auditing in fault diagnosis / condition monitoring.
3. Causal discovery or causal graph use in mechanical fault diagnosis.
4. PC/PC-KCI, causal adjacency, and limitations of causal direction interpretation in observational data.
5. Wavelet/time-frequency representations for gearbox/bearing diagnosis.
6. Trustworthy/reliable AI for rotating machinery and industrial PHM.
7. Multi-view explanation: sensor/channel view and frequency/time-scale view.
8. Papers combining attribution with causal discovery, or comparing attribution and causal graph structure.

## Academic Questions To Decide

- Is the strongest novelty the SHAP-to-dual-view-PC-DAG bridge, the attribution-graph consistency audit, or shortcut-risk screening?
- Is "causal" acceptable in the title, or should the wording use "graph consistency" / "causality-adjacency" to stay conservative?
- How much baseline performance is required for a high-level journal if the main focus is diagnostic logic audit?
- Should attention, CWRU, GNN, and cross-condition be supplement/future only?
- Which 1-2 case studies best demonstrate the method without overclaiming?

## Follow-Up Experiments To Design

P0:

- Recover collaborator baseline numeric metrics, or define a formal rerun protocol later.
- Freeze main SHAP + PC-DAG consistency tables.
- Select main case study.
- Freeze PC-DAG edge policy and graph figure settings.

P1:

- Structured classification report / confusion matrix table.
- SHAP top-k table polishing.
- Publication-quality SHAP/DAG/case figures.
- Config/provenance manifest for baseline/SHAP/PC-DAG.

P2:

- CWRU external validation if metrics can be recovered.
- Noise robustness / missing-channel robustness.
- GNN repair and verification only if a future extension is desired.
- Clean cross-condition rerun only as support, not current main evidence.

## Do Not Drift

- Do not treat the local crop-width reproduction gap as proof that collaborator results failed.
- Do not force GNN, CWRU, or cross-condition into the main contribution.
- Do not overclaim causal direction; use strict feature_to_label only for directional support.
- Do not write the paper as a generic high-accuracy classifier paper.
- Do not present Codex's extraction/audit scripts as the collaborator's original scientific method.
- Keep the paper aligned with the collaborator's original assets and likely intent.

## Requested Deep Research Output Structure

Please produce a report with:

1. Recommended SCI Q1 positioning and target-journal style.
2. Final main storyline and 3-5 contribution bullets.
3. Claim-evidence matrix with conservative wording.
4. Required figures/tables and what each should show.
5. Minimal follow-up experiment plan with P0/P1/P2.
6. Literature map and how this work differs from prior SHAP/XAI/causal-diagnosis papers.
7. Risks, limitations, and wording constraints.
8. Questions for the human researcher/collaborator before drafting.
