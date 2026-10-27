#!/bin/bash
cd "$(dirname "$0")"; export PYTHONWARNINGS=ignore; OUT=results/diag
run() { tag=$1; shift; python -u train.py --tag $tag --backbone sasrec --use_id --item_dim 64 --epochs 60 --patience 60 --dropout 0.1 --out_dir $OUT "$@" > $OUT/$tag.log 2>&1; }
run cos_len50  --max_len 50
run cos_len200 --max_len 200
echo DONE > $OUT/DONE
