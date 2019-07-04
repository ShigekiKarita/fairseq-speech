# coding: utf-8

import torch

from fssp.features import ASRFeature


def test_conv_fbank():
    x = torch.sin(torch.arange(0, 2000).float())
    feature = ASRFeature(80, n_subsample=2)
    y = feature(x.unsqueeze(0))
    new_len = feature.length(torch.LongTensor([x.shape[0]]))
    assert y.shape == (1, new_len[0], 80)

