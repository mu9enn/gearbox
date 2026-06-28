# Baseline Metrics Notes

- Evaluated checkpoints: 20.
- Successful metric rows: 20.
- Splits evaluated: test.
- Crop width: 512.
- Historical width caveat: true.
- Human researcher confirmed crop assumption: true.
- Note: old checkpoint/SHAP used 512-width input; current wavelet data are 1024-width; human researcher confirmed using first 512 points via [..., :512] to recover historical evaluation protocol

## Paper Use

- These rows are candidate Table 1 baseline metrics only under the recorded historical-width caveat.
- Cross-condition splits are not evaluated here and should not be inferred from this table.
