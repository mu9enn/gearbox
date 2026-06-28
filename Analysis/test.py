import numpy as np


path = '../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz'
data = np.load(path)
print(data.keys())
print(data['weights'])