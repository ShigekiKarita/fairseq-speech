#!/usr/bin/env python3
import argparse

import kaldiio
import torch

from fssp.datasets import wav2float
from fssp.features import ASRFeature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scp", nargs="+")
    parser.add_argument("--out", help="output file")
    parser.add_argument("--dim", type=int, help="fbank dim")
    args = parser.parse_args()

    feat = ASRFeature(odim=args.dim, n_subsample=0)

    mean = torch.zeros(args.dim)
    mean2 = torch.zeros(args.dim)
    tsum = 0
    for scp in args.scp:
        n = len(kaldiio.load_scp(scp))
        d = kaldiio.load_scp_sequential(scp)
        for i, (k, v) in enumerate(d):
            print(f"progress: {i} / {n}")
            # TODO(karita) batch processing
            sr, wav = v
            wav = torch.from_numpy(wav2float(wav))
            fbank = feat(wav.unsqueeze(0))[0]
            t = fbank.shape[0]
            mean = (t * fbank.sum(0) + tsum * mean) / (t + tsum)
            mean2 = (t * (fbank * fbank).sum(0) + tsum * mean2) / (t + tsum)

    var = mean * mean - mean2
    d = dict(mean=mean, stddev=var.sqrt(), var=var)
    print(d)
    torch.save(d, args.out)


if __name__ == "__main__":
    main()
