# Checkpoint Dataset Mismatch Notes

## Findings

- Checkpoints inspected: 20.
- Checkpoint expected widths: {'512': 20}.
- Current dataset test widths: {'1024': 20}.
- SHAP cache widths: {'512': 20}.
- Current `wavelet_dataset/*.npz` files contain `train_set/valid_set/test_set` arrays with width 1024.
- Existing NoAttention SHAP caches have shape `(500, 6, 6, 512, 5)`, so they were produced from width-512 model inputs.
- Legacy training/SHAP scripts reference lowercase dataset names and keys such as `train_data/test_data/class_names`, which are not present in the current uppercase `wavelet_dataset/*.npz` contract.

## Safety Judgment

- It is technically possible to run the saved checkpoints on `current_split[..., :512]` because the checkpoint architecture expects width 512.
- However, this repository does not currently contain the original lowercase 512-width `.npz` files, nor explicit training-code evidence that the old data was exactly the first 512 points of the current 1024-point arrays.
- Therefore `--crop-width 512` should be treated as a compatibility/smoke evaluation, not as paper-grade baseline metrics, unless human researcher confirms the old preprocessing contract.

## Script Evidence

### Explicit 512 / crop-related hits
- `Common/NetWorkFrame.py:144: nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3),padding=(1, 1), stride=(1, 1)),#(b,6,6,512)`

### Legacy dataset key/path hits
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:37: wavelet_path = "../../../wavelet_dataset/bearing_20_0.npz"`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:39: train_data = wavelet_datasets['train_data']`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:41: test_data = wavelet_datasets['test_data']`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:43: class_num = len(wavelet_datasets['class_names'])`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:46: bearing_20_0_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in BearingLabel_20_0]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:48: train_data_mask = np.isin(train_labels, bearing_20_0_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:49: test_data_mask = np.isin(test_labels, bearing_20_0_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:50: train_data = train_data[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:51: train_labels = train_labels[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:52: test_data = test_data[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:53: test_labels = test_labels[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:55: "train_data": train_data,`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_20_0.py:57: "test_data": test_data,`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:39: train_data = wavelet_datasets['train_data']`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:41: test_data = wavelet_datasets['test_data']`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:43: class_num = len(wavelet_datasets['class_names'])`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:46: bearing_30_2_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in BearingLabel_30_2]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:48: train_data_mask = np.isin(train_labels, bearing_30_2_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:49: test_data_mask = np.isin(test_labels, bearing_30_2_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:50: train_data = train_data[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:51: train_labels = train_labels[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:52: test_data = test_data[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:53: test_labels = test_labels[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:55: "train_data": train_data,`
- `ModelTrain/NoAttention/model_train_6ch/Bearing_30_2.py:57: "test_data": test_data,`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:37: wavelet_path = "../../../wavelet_dataset/gear_20_0.npz"`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:39: train_data = wavelet_datasets['train_data']`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:41: test_data = wavelet_datasets['test_data']`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:43: class_num = len(wavelet_datasets['class_names'])`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:46: gear_20_0_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in GearLabel_20_0]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:48: train_data_mask = np.isin(train_labels, gear_20_0_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:49: test_data_mask = np.isin(test_labels, gear_20_0_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:50: train_data = train_data[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:51: train_labels = train_labels[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:52: test_data = test_data[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:53: test_labels = test_labels[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:55: "train_data": train_data,`
- `ModelTrain/NoAttention/model_train_6ch/Gear_20_0.py:57: "test_data": test_data,`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:39: train_data = wavelet_datasets['train_data']`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:41: test_data = wavelet_datasets['test_data']`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:43: class_num = len(wavelet_datasets['class_names'])`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:46: gear_30_2_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in GearLabel_30_2]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:48: train_data_mask = np.isin(train_labels, gear_30_2_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:49: test_data_mask = np.isin(test_labels, gear_30_2_labels)`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:50: train_data = train_data[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:51: train_labels = train_labels[train_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:52: test_data = test_data[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:53: test_labels = test_labels[test_data_mask]`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:55: "train_data": train_data,`
- `ModelTrain/NoAttention/model_train_6ch/Gear_30_2.py:57: "test_data": test_data,`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:26: wavelet_path = '../../wavelet_dataset/bearing_20_0.npz'`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:29: # print(wavelet_datasets['class_names'])`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:30: train_data = wavelet_datasets['train_data']`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:32: test_data = wavelet_datasets['test_data']`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:34: class_num = len(wavelet_datasets['class_names'])`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:35: # print(train_data.shape[0] + test_data.shape[0])`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:36: # print(train_data.shape)`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:37: # print(test_data.shape)`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:82: train_data = core_channel_selection(train_data, index=range(6))`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:83: test_data = core_channel_selection(test_data, index=range(6))`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:87: bearing_20_0_fault_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in BearingLabel_20_0]`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:89: back_label = np.where(wavelet_datasets['class_names'] == 'bearing_normal_20_0')[0][0]`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:94: back_train = train_data[background_sample_mask_train]`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:95: back_test = test_data[background_sample_mask_test]`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:112: every_fault_sample_train = train_data[mask_train]`
- `CalculateShapValues/NoAttention/Bearing_20_0.py:113: every_fault_sample_test = test_data[mask_test]`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:29: # print(wavelet_datasets['class_names'])`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:30: train_data = wavelet_datasets['train_data']`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:32: test_data = wavelet_datasets['test_data']`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:34: class_num = len(wavelet_datasets['class_names'])`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:35: # print(train_data.shape[0] + test_data.shape[0])`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:36: # print(train_data.shape)`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:37: # print(test_data.shape)`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:82: train_data = core_channel_selection(train_data, index=range(6))`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:83: test_data = core_channel_selection(test_data, index=range(6))`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:87: bearing_30_2_fault_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in BearingLabel_30_2]`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:89: back_label = np.where(wavelet_datasets['class_names'] == 'bearing_normal_30_2')[0][0]`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:94: back_train = train_data[background_sample_mask_train]`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:95: back_test = test_data[background_sample_mask_test]`
- `CalculateShapValues/NoAttention/Bearing_30_2.py:112: every_fault_sample_train = train_data[mask_train]`
