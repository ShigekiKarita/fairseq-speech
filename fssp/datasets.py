# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import os
import numpy as np
import sys
import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset, data_utils
import kaldiio
import torch
from torch.nn.utils.rnn import pad_sequence


# TODO(karita) lazy load
def load_text(path, tgt_dict):
    d = dict()
    sizes = []
    with open(path, "r") as f:
        for line in f:
            xs = line.split()
            d[xs[0]] = torch.LongTensor([tgt_dict.index(x) for x in xs[1:]] + [tgt_dict.eos()])
            sizes.append(len(xs) - 1)
    return d, sizes


def collate(
    samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        # returns (B, maxlen)
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    # NOTE: only changed this line
    src_tokens = pad_sequence([torch.tensor(s['source']).float() for s in samples], batch_first=True)
    # sort by descending source length
    src_lengths = torch.LongTensor([torch.tensor(s['source']).numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,               # torch.LongTensor
        'nsentences': len(samples),
        'ntokens': ntokens,     # int
        'net_input': {
            'src_tokens': src_tokens,    # torch.FloatTensor(B, max_src_len, idim)
            'src_lengths': src_lengths,  # torch.LongTensor(B)
        },
        'target': target,  # torch.LongTensor(B, max_tgtlen)
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def wav2float(wav, dtype=np.float32):
    return wav.astype(dtype) / np.iinfo(wav.dtype).max


class ASRDataset(FairseqDataset):

    def __init__(self, wav_scp_path, text_path, tgt_dict, shuffle=False):
        super().__init__()
        self._sample_rate = None
        self.wav_dict = kaldiio.load_scp(wav_scp_path)
        self.text_dict, self.sizes = load_text(text_path, tgt_dict)
        self.id2key = list(self.text_dict.keys())
        self.tgt_dict = tgt_dict
        self.shuffle = shuffle

    @property
    def sample_rate(self):
        return self._sample_rate

    def __getitem__(self, index):
        if isinstance(index, str):
            key = index
        else:
            key = self.id2key[index]

        sr, wav = self.wav_dict[key]
        wav = wav2float(wav)
        if self.sample_rate is None:
            self._sample_rate = sr
        elif sr != self.sample_rate:
            import logging
            logging.warning(f"resample from {sr} to {self.sample_rate}")
            factor = self.sample_rate / sr
            wav = self.resample(wav, factor)

        return {
            'id': index,
            'key': key,
            'source': wav,
            'target': self.text_dict[key]
        }

    def resample(self, x, factor):
        return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

    def __len__(self):
        return len(self.id2key)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.tgt_dict.pad(), eos_idx=self.tgt_dict.eos(),
        )

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]  # TODO(karita):  source length

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)
