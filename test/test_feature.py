# coding: utf-8

import torch

import fssp.feature

x = torch.sin(torch.arange(0, 2000).float())
feature = fssp.feature.ASRFeature(80, n_subsample=2)
y = feature(x.unsqueeze(0))
print(y.shape)
