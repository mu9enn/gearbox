# GNN Risk Report

## GNNCausalSEU.forward

- Computes causal propagation variables: yes
- Final classifier input detected: `x`
- Risk: current SEU GNN forward pass does not support a strong `causal-enhanced classification` claim, because the classifier is fed `x` rather than propagated `out`.

## GNNCausal/CausalGNN.py scaler usage

- Scaler consistency assessment: `inconsistent_or_split_refit`
- `fit_transform` calls:
  - `GNNCausal/CausalGNN.py:73: train_sample = scaler.fit_transform(train_sample)`
  - `GNNCausal/CausalGNN.py:76: finetune_sample = scaler.fit_transform(finetune_sample)`
- `transform` calls:
  - `GNNCausal/CausalGNN.py:74: valid_sample = scaler.transform(valid_sample)`
  - `GNNCausal/CausalGNN.py:75: test_sample = scaler.transform(test_sample)`
  - `GNNCausal/CausalGNN.py:77: cross_test_sample = scaler.transform(cross_test_sample)`
- Risk: finetune/cross branch appears to refit a scaler instead of using the train-fitted scaler consistently.
