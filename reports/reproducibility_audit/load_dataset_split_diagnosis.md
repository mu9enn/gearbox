# Load Dataset Split Diagnosis

## 1. Current Split Generation

- `Common/LoadDatasets.py` reads SEU CSV files with `pd.read_csv(skiprows=16, usecols=range(8), sep=r'\s+|,')`.
- It selects vibration channels `[1, 2, 3, 5, 6, 7]` for each fault-condition sequence.
- For ordinary splits, each original continuous sequence is cut by time order before angle resampling:
  - `train`: first 80% raw rows.
  - `valid`: next 10% raw rows.
  - `test`: final 10% raw rows.
- Each split is angle-resampled independently, then windowed:
  - `train_set`: `window_data(..., step=768)`.
  - `valid_set`: `window_data(..., step=1024)`.
  - `test_set`: `window_data(..., step=768)`.
- Each window is transformed by SWT with `db4`, `layers=5`, producing `N x 6 x 6 x 1024`.
- `cross_finetune_set` is generated differently: a random contiguous 10% raw segment is sampled from the full original sequence.
- `cross_test_set` is then defined as all remaining raw rows outside that random 10% segment.

## 2. Most Likely Overlap Source

- The strongest leakage source is `cross_test_set = full_sequence - cross_finetune_segment`.
- Because `cross_test_set` is built from the same full original sequence, it normally includes large parts of the original `train`, `valid`, and `test` raw intervals unless the random finetune segment happens to cover them.
- Therefore `train_set` and `cross_test_set` are not independent evaluation splits under the current contract.

## 3. Exact Duplicate vs Same-Source Adjacent-Window Risk

- Full sample hashing can confirm exact duplicate windows after wavelet transform.
- Even when exact duplicates are absent, current `cross_test_set` still has same-source leakage risk because it reuses the same continuous sequence that produced ordinary train/valid/test.
- The ordinary `train/valid/test` split is less severe because it uses contiguous raw blocks, but `step=768` with `segment_length=1024` creates overlapping windows inside train/test. This is acceptable only within a split, not across evaluation boundaries.

## 4. Historical Reference Splits

- Existing `train/valid/test` can be retained as historical within-condition reference only.
- Existing `cross_finetune_set` and `cross_test_set` should be retained only as historical/debug reference.

## 5. Splits Not Suitable for Paper Evidence

- Existing cross-condition or finetune/cross-test evidence derived from current `cross_finetune_set` / `cross_test_set` is not paper-grade.
- Any result depending on `cross_test_set` as an independent cross-domain target should be regenerated after a strict no-overlap data contract is established.

## 6. Recommended Leak-Free Redefinition

- Split each original fault-condition CSV into mutually exclusive raw contiguous blocks before angle resampling, windowing, or wavelet transform.
- Use dedicated raw blocks for `train`, `valid`, `test`, `cross_finetune`, and `cross_test`.
- Add raw guard gaps between blocks to reduce adjacent-window boundary leakage.
- Use non-overlapping windows for the leak-free dataset unless a later experiment explicitly studies overlapping-window augmentation.
- Persist a manifest containing source files, split policy, output shapes, sample hash summaries, git commit, and known limitations.
