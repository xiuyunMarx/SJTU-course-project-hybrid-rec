#!/bin/bash
# 短训练扫描：SASRec 宽度 和 GPT-2 学习率。日志写到 results/sweep/。
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
OUT=results/sweep
for d in 64 768; do
  python -u train.py --tag sasrec_id_d$d --backbone sasrec --use_id --item_dim $d --epochs 15 --patience 15 --out_dir $OUT > $OUT/sasrec_id_d$d.log 2>&1
done
for lr in 5e-5 1e-4 3e-4; do
  python -u train.py --tag gpt2_text_lr$lr --backbone gpt2 --use_text --lr $lr --epochs 10 --patience 10 --out_dir $OUT > $OUT/gpt2_text_lr$lr.log 2>&1
done
python -u train.py --tag gpt2_scratch_lr3e-4 --backbone gpt2 --use_text --no_pretrained --lr 3e-4 --epochs 10 --patience 10 --out_dir $OUT > $OUT/gpt2_scratch_lr3e-4.log 2>&1
echo SWEEP_DONE > $OUT/DONE
