# Figure Selection Notes

- Existing figures were selected without redrawing or rerunning SHAP/PC-DAG.
- Fig1 should be drawn by the author because no polished framework overview exists.
- PC-DAG figures should be described under the conservative edge policy: feature_to_label is strict support; label_to_feature and undirected edges are weak adjacency only.
- Shortcut-risk cases are candidates, not confirmed shortcuts.
- Baseline supplement inherits the historical-width caveat: old checkpoint/SHAP used 512-width input; current wavelet data are 1024-width; human researcher confirmed using first 512 points via [..., :512] to recover historical evaluation protocol.
