#!/bin/bash
# miniHLLM 正式实验：6 个模型 × 3 个种子，序列长 100、最多 200 轮、耐心 20。
# 用法：./run_all.sh            # 顺序执行 A 再 B（GPT-2 单个进程约 21G 显存，不能并行）
#       ./run_all.sh A | B      # 只跑其中一条
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/main
COMMON="--max_len 100 --epochs 200 --patience 20 --out_dir $OUT"
SEEDS="42 2026 3407"

run() {  # run <tag> <seed> <args...>
  tag=$1; seed=$2; shift 2
  [ -f $OUT/${tag}_seed${seed}.json ] && return   # 已有结果则跳过
  ckpt=""; [ "$seed" = "42" ] && ckpt="--save_ckpt"
  python -u train.py --tag $tag --seed $seed $COMMON $ckpt "$@" > $OUT/${tag}_seed${seed}.log 2>&1
}

stream_A() {
  run popularity 42 --model popularity
  for s in $SEEDS; do run sasrec_id      $s --backbone sasrec --use_id            --item_dim 64 --dropout 0.1 --lr 1e-3; done
  for s in $SEEDS; do run sasrec_id_text $s --backbone sasrec --use_id --use_text --item_dim 64 --dropout 0.1 --lr 1e-3; done
  for s in $SEEDS; do run hllm_text      $s --backbone gpt2 --use_text --lr 1e-4; done
  echo DONE > $OUT/DONE_A
}
stream_B() {
  for s in $SEEDS; do run hllm_text_scratch $s --backbone gpt2 --use_text --no_pretrained --lr 3e-4; done
  for s in $SEEDS; do run hllm_text_id      $s --backbone gpt2 --use_text --use_id --lr 1e-4; done
  echo DONE > $OUT/DONE_B
}

case "${1:-both}" in
  A) stream_A ;;
  B) stream_B ;;
  *) stream_A; stream_B ;;
esac
