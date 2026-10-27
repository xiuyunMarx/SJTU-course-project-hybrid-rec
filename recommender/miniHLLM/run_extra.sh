#!/bin/bash
# 补充实验：SASRec 纯 ID 400 轮上限、SASRec 仅文本、冻结 GPT-2、随机初始化同学习率对照。
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/main
COMMON="--max_len 100 --patience 20 --out_dir $OUT"
SEEDS="42 2026 3407"
run() { tag=$1; seed=$2; shift 2
  [ -f $OUT/${tag}_seed${seed}.json ] && return
  ckpt=""; [ "$seed" = "42" ] && ckpt="--save_ckpt"
  python -u train.py --tag $tag --seed $seed $COMMON $ckpt "$@" > $OUT/${tag}_seed${seed}.log 2>&1; }

for s in $SEEDS; do run sasrec_id_400ep        $s --backbone sasrec --use_id            --item_dim 64 --dropout 0.1 --lr 1e-3 --epochs 400; done
for s in $SEEDS; do run sasrec_text            $s --backbone sasrec --use_text          --item_dim 64 --dropout 0.1 --lr 1e-3 --epochs 200; done
for s in $SEEDS; do run hllm_text_frozen       $s --backbone gpt2 --use_text --freeze_backbone --lr 1e-3 --epochs 200; done
for s in $SEEDS; do run hllm_text_scratch_lr1e-4 $s --backbone gpt2 --use_text --no_pretrained --lr 1e-4 --epochs 200; done
echo DONE > $OUT/DONE_EXTRA
