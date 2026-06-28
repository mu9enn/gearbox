import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import os

# def load_and_split_pairs(causal_edge:torch.Tensor=None) -> list:
#
#     causal_set = set()
#     for u, v in zip(causal_edge[0].tolist(), causal_edge[1].tolist()):
#         causal_set.add((u, v))
#         causal_set.add((v, u))
#
#     # 生成所有无向节点对 (u < v)，去重
#     all_pairs = list(combinations(range(18), 2))
#     # 筛选：不在因果集合中的 = 非因果对
#     non_causal_pairs = [pair for pair in all_pairs if pair not in causal_set]
#     return non_causal_pairs

arr = list(combinations(range(18), 2))
print(arr)
print(len(arr))
