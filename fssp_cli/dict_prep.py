#!/usr/bin/env python3
import sys


def get_parser():
    import argparse
    parser = argparse.ArgumentParser("dictionary ")
    parser.add_argument("--token", help="tokenizer mode", default="word", choices=["word"])
    return parser


args = get_parser().parse_args()
vocab = set()
for line in sys.stdin:
    xs = line.split()
    for x in xs[1:]:
        vocab.add(x)

for i, v in enumerate(sorted(list(vocab))):
    print(v, i)
