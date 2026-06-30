# Paper Storyline Candidates

This file is planning material for ChatGPT and the human researcher. It is not manuscript text.

## Storyline A: Attribution-Graph Consistency Audit For Trustworthy Rotating Machinery Diagnosis

Recommended.

Main problem:

- A CNN fault diagnosis model can produce predictions, but the paper asks whether its learned attribution logic is consistent with feature-label graph structure.

Core contribution:

- Build an audit chain from wavelet tensor diagnosis to SHAP channel/frequency attribution, then to SHAP-weighted dual-view PC-DAG and conservative attribution-graph consistency analysis.
- Use strict `feature_to_label` edges only for directional support.
- Use `label_to_feature` and undirected label-related edges only as weak label adjacency / graph-consistency evidence.
- Identify high-SHAP graph-unsupported features as shortcut-risk candidates and low-SHAP graph-supported features as underused candidates.

Existing support:

- SHAP cache and figures: `CalculateShapValues/NoAttention/`.
- SHAP seed stability: `reports/paper_evidence/shap_seed_stability_summary.csv`.
- PC inputs: `PC_Datasets/Seed_49/` and `PC_Datasets/average/`.
- PC-DAG figures: `DAG/PC_DAG/Seed_49/`.
- Edge and consistency tables: `reports/paper_evidence/pcdag_edges/` and `reports/paper_evidence/consistency/`.
- Candidate cases: `reports/paper_evidence/final_tables/`.

Minimum missing pieces:

- Recover collaborator baseline numeric metrics or plan a formal rerun later.
- Freeze final PC-DAG alpha/filtered/average figure choices.
- Select 1-2 case studies and prepare clean main figures.

Risks:

- Strict directional support is sparse.
- Baseline local crop-width eval is a provenance diagnostic only, not a paper result.
- Cross-condition and GNN should not enter the main contribution.

## Storyline B: Interpretable Diagnostic Pipeline With SHAP And Dual-View Causal Graph

More engineering/application oriented.

Main problem:

- Build an interpretable diagnostic pipeline for rotating machinery where attribution and graph structure jointly explain model behavior.

Core contribution:

- Present a practical workflow: wavelet tensor -> CNN/no-attention diagnosis -> SHAP explanation -> channel/frequency PC-DAG -> graph-consistency audit.
- Use attention comparison, confusion/F1 plots, and possibly CWRU assets as supporting material if numeric results can be recovered.

Existing support:

- NoAttention and Attention/NoAttention result directories under `ModelTrain/`.
- SHAP figures and stability tables.
- PC-DAG figures and edge summaries.
- Existing confusion matrix/F1 figures for supplement.

Minimum missing pieces:

- Trustworthy baseline metrics table from collaborator records or a later formal rerun.
- Structured classification reports if confusion/F1 are used beyond visual supplement.
- Decide whether attention stays supplement-only.

Risks:

- This framing may dilute the sharper paper contribution by inviting baseline/attention/CWRU claims that are not yet structurally ready.
- Reviewers may expect more complete model-performance comparisons.

## Storyline C: Case-Study Based Diagnostic Logic Audit Framework

Most conservative.

Main problem:

- Demonstrate a reusable diagnostic logic audit framework through representative gearbox/bearing case studies rather than broad performance claims.

Core contribution:

- Show how SHAP and dual-view PC-DAG can expose consistent, conflicting, shortcut-risk, and underused feature patterns.
- Emphasize methodology and case evidence rather than broad generalization.

Existing support:

- Strongest strict case: `Bearing_30_2 / frequency / cD2`.
- Weak adjacency case: `Bearing_20_0 / frequency / cD2`.
- Shortcut-risk candidate: `Gear_20_0 / channel / ch8` or `Gear_20_0 / frequency / cD4`.
- Underused graph-supported candidate: `Bearing_20_0 / channel / ch3`.

Minimum missing pieces:

- Baseline numeric context still helpful, but less central than in Storyline B.
- Clean figure panels for each selected case.

Risks:

- More conservative and possibly narrower for SCI unless the discussion strongly explains why logic-audit evidence matters.
- Needs careful language to avoid seeming like only post-hoc visualization.
