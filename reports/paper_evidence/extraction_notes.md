# Paper Evidence Extraction Notes

## Extracted

- Baseline asset inventory rows: 191.
- Baseline metric rows: 191; numeric fields are mostly unavailable because existing assets are figures/checkpoints, not metrics CSV.
- SHAP rank rows: 220; seed stability rows: 44.
- PC-DAG asset inventory rows: 230.
- Consistency candidate rows: 44.

## Missing Or Limited

- Existing baseline curves/confusion/F1 are mostly PNG-only; no OCR or image-value inference was performed.
- Existing DAG figures are available, but learned edge tables are not persisted.
- Consistency candidates therefore contain reliable SHAP-side ranks but unknown causal-edge columns.
- Cross-condition assets are marked risk/reference only because current cross split has train/cross_test exact duplicates.
- GNN assets are marked risk/reference only because GNNCausalSEU currently feeds `x` rather than propagated `out` into the classifier.

## SHAP Notes

- Aggregation follows existing NoAttention scripts: mean absolute SHAP over samples, non-target axes, time, and class outputs.
- Frequency extraction uses `shap_values[:,:,0:-1,:,:]` and names the five bands as `cA5,cD5,cD4,cD3,cD2`, matching `PCDataset(wavelet).py` and `DAG/WaveletDAG.py`.

## Next Minimal Engineering Step

- Add metric export to evaluation/confusion scripts without retraining.
- Add edge CSV export to PC-DAG generation next time PC-KCI is run.
- Join exported PC-DAG edges with `shap_seed_stability_summary.csv` to finalize the consistency/conflict audit table.
