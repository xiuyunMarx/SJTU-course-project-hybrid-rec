#!/bin/bash
# 补充扫描：GPT-2 骨干下 点积 vs 余弦；以及序列长 200 时的每轮耗时。
cd "$(dirname "$0")"; export PYTHONWARNINGS=ignore; OUT=results/sweep2
python -u train.py --tag gpt2_dot_len50  --backbone gpt2 --use_text --scoring dot --lr 1e-4 --epochs 10 --patience 10 --out_dir $OUT > $OUT/gpt2_dot_len50.log 2>&1
python -u train.py --tag gpt2_cos_len200 --backbone gpt2 --use_text --lr 1e-4 --max_len 200 --epochs 3 --patience 3 --out_dir $OUT > $OUT/gpt2_cos_len200.log 2>&1
echo DONE > $OUT/DONE
