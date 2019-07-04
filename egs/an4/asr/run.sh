#!/bin/bash

# data
stage=${1:--1}
datadir=./downloads
an4_root=${datadir}/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/

token=sentencepiece # sentencepiece
bpe_size=100
fbank=80

if [ $token = "sentencepiece" ]; then
    bpe_model=data/train/bpe.model
    suffix=.bpe
    dict=${bpe_model}.vocab
else
    suffix=""
    dict=data/train/dict.${token}.txt
fi

FSSP=../../..
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
        cut -d" " -f2- ./data/${x}/text > ./data/${x}/raw_text
    done

    if [ $token = "sentencepiece" ]; then
        python $FSSP/fssp_cli/spm_train.py \
               --input=./data/train/raw_text \
               --model_prefix=$bpe_model \
               --vocab_size=$bpe_size \
               --character_coverage=1.0 \
               --model_type=bpe

        for src in ./data/*/text; do
            raw=$(dirname $src)/raw_text
            cut -d" " -f2- $src > $raw
            python $FSSP/fssp_cli/spm_encode.py \
                   --model ${bpe_model}.model \
                   --output_format=piece \
                   --inputs $raw \
                   --outputs ${raw}.bpe

            paste -d' '  <(cut -d' ' -f1 $src) ${raw}.bpe > $src.bpe
        done
    else
        $FSSP/fssp_cli/dict_prep.py --token ${token} < data/train/text > ${dict}
        # fairseq-preprocess --only-source \
        #                    --trainpref ./data/train/raw_text \
        #                    --validpref ./data/test/raw_text \
        #                    --destdir data/bin \
        #                    --workers 10
    fi

    $FSSP/fssp_cli/cmvn.py data/train/wav.scp --out data/train/cmvn.pt --dim ${fbank}
fi

fairseq-train --user-dir $FSSP/fssp -a asr_transformer --task asr \
              --wav ./data/*/wav.scp --text ./data/*/text${suffix} \
              --dict ./data/train/dict.txt --cmvn ./data/train/cmvn.pt \
              --train-subset train --valid-subset test \
              --fbank-dim ${fbank} --num-workers 8  \
              --encoder-layers 3 --decoder-layers 3 \
              --max-tokens 800 \
              --optimizer adam --lr 1e-2 --clip-norm 10.0 \
              --label-smoothing 0.1 --dropout 0.1 \
              --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
              --criterion cross_entropy_with_accuracy --max-update 50000 \
              --warmup-updates 4000 --warmup-init-lr '1e-07' \
              --adam-betas '(0.9, 0.98)' # --save-dir checkpoints/transformer
