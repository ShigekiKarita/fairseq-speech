#!/bin/bash

# data
stage=${1:--1}
datadir=./downloads
an4_root=${datadir}/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/
dict=data/train/dict.txt
fbank=80

FSSP=../../../
PYTHONPATH=$FSSP

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${datadir}
    local/download_and_untar.sh ${datadir} ${data_url}
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    mkdir -p data/{train,test} exp

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python local/data_prep.py ${an4_root} ${datadir}/sph2pipe_v2.5/sph2pipe

    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
    done

    python local/dict_prep.py < data/train/text  > ${dict}
    $FSSP/bin/fssp-cmvn.py data/train/wav.scp --out data/train/cmvn.pt --dim ${fbank}
fi

fairseq-train --user-dir $FSSP/fssp -a asr_transformer --task asr \
              --wav ./data/*/wav.scp --text ./data/*/text \
              --dict ./data/train/dict.txt --cmvn ./data/train/cmvn.pt \
              --train-subset train --valid-subset test \
              --fbank-dim ${fbank} \
              --max-tokens 300 \
              --optimizer adam --lr 1e-4 --clip-norm 0.1 \
              --label-smoothing 0.1 --dropout 0.1 \
              --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
              --criterion label_smoothed_cross_entropy --max-update 50000 \
              --warmup-updates 4000 --warmup-init-lr '1e-07' \
              --adam-betas '(0.9, 0.98)' # --save-dir checkpoints/transformer
