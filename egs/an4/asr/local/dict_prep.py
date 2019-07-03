import sys


vocab = set()
for line in sys.stdin:
    xs = line.split()
    for x in xs[1:]:
        vocab.add(x)

for i, v in enumerate(sorted(list(vocab))):
    print(v, i)
