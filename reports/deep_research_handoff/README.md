
# Deep Research Handoff Package

This directory is the final local-facts handoff from Codex to ChatGPT before deep literature research and SCI paper planning.

Recommended reading order:

1. `README.md` - this guide.
2. `01_author_contribution_reconstruction.md` - reconstructed collaborator contribution and original-intent interpretation.
3. `02_local_experiment_result_digest.csv` - asset/result inventory, including main/supplement/risk/future classifications.
4. `03_claim_evidence_gap_matrix.csv` - candidate paper claims, evidence, gaps, and safe wording level.
5. `04_main_case_study_asset_pack.csv` - concrete case candidates for the main story and figures.
6. `05_figure_table_materials_pack.csv` - proposed figures/tables and available assets.
7. `06_followup_experiment_blueprint.csv` - P0/P1/P2 follow-up tasks for a high-level SCI paper.
8. `07_original_author_intent_questions.md` - questions that should be asked of the collaborator/human researcher.
9. `08_chatgpt_deep_research_prompt.md` - prompt for ChatGPT's next-stage deep research report.

Important caveats:

- The local `--crop-width 512` evaluation is diagnostic only and not paper evidence.
- The strongest current mainline is SHAP attribution plus dual-view PC-DAG attribution-graph consistency audit.
- GNN, CWRU, cross-condition, and attention should not be forced into the main contribution unless the human researcher later decides otherwise.
- Causal wording must stay conservative: `feature_to_label` is strict support; `label_to_feature` and undirected label-related edges are weak adjacency only.
