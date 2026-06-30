
# Original Author / Collaborator Intent Questions

These questions are intended for the human researcher or collaborator. They focus only on issues that can change the paper evidence chain.

## Baseline And Data Provenance

1. Where are the original baseline numeric result tables for the NoAttention CNN? Are they in logs, notebooks, spreadsheets, or only figures?
2. Which model family, seed set, and condition set were intended as the final diagnostic baseline in the original work?
3. What is the historical relationship between the 512-width checkpoint/SHAP input and the current 1024-width `wavelet_dataset` files?
4. Which data split contract should be considered the official one for the original paper: historical lowercase `train_data/test_data`, current uppercase `train_set/test_set`, or another archive?
5. Were the existing confusion/F1 figures produced from the same checkpoints used for SHAP?

## SHAP And Attribution

6. Which checkpoints exactly correspond to the existing SHAP caches under `CalculateShapValues/NoAttention/Seed_*`?
7. Were SHAP background and explanation samples chosen from train/test intentionally, and how should this be described?
8. Were channel and frequency SHAP rankings intended as final quantitative results or mainly as visualization?
9. Should the paper emphasize channel attribution, frequency attribution, or the contrast between both?

## PC-DAG And Causal Graphs

10. Which PC-DAG setting was intended for the final paper: Seed_49, average, alpha 0.05, alpha 0.001, filtered, original, or another combination?
11. How did the original author intend to interpret `label_to_feature` edges? Should they be discarded, treated as adjacency, or discussed as orientation instability?
12. Is the physical prior/background knowledge meant to strictly forbid label -> feature in the final graph, or only guide graph visualization?
13. Should the final claim use causal-direction language or safer graph-consistency/audit language?
14. Were shortcut-risk candidates an intended original contribution, or are they a post-hoc framing of attribution-graph conflicts?

## Auxiliary Branches

15. Was GNN intended as a main contribution, a future extension, or only an exploratory experiment?
16. If GNN was intended as a contribution, should `GNNCausalSEU.forward` classify from propagated features rather than the raw `x` tensor?
17. Was CWRU intended as external validation for this paper or as a separate follow-up?
18. Were cross-condition results intended for the paper despite the current split-overlap risk?
19. Are noise/pure-data assets part of a planned robustness experiment or only signal-processing exploration?

## Paper Framing

20. What should the paper primarily emphasize: diagnosis performance, interpretability, causal/graph audit, shortcut-risk screening, or engineering application?
21. Which 1-2 conditions/cases does the collaborator consider most convincing?
22. Which figures were originally planned as main figures?
23. Are there unpublished slides, thesis figures, spreadsheets, or logs that better capture the original result story?
24. What claim would the collaborator most want reviewers to remember?
