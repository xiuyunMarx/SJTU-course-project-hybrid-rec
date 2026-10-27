"""miniHLLM 训练与评测入口。

示例：
  python train.py --tag sasrec_id   --backbone sasrec --use_id
  python train.py --tag hllm        --backbone gpt2  --use_text
  python train.py --tag hllm_scratch --backbone gpt2 --use_text --no_pretrained
  python train.py --tag popularity  --model popularity
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models import GPT2Backbone, ItemEncoder, PopularityRecommender, SASRecBackbone, SeqRecommender
from seq_data import SeqData

HERE = Path(__file__).resolve().parent          # recommender/miniHLLM
ROOT = HERE.parents[1]                          # 仓库根目录


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, data):
    if args.model == "popularity":
        return PopularityRecommender(data.item_pop, data.valid_item_mask)

    text = None
    if args.use_text:
        feats = torch.load(args.features, map_location="cpu")
        text = feats[args.pooling]
    item_dim = args.item_dim
    item_encoder = ItemEncoder(data.n_items, item_dim, text, use_text=args.use_text, use_id=args.use_id)

    if args.backbone == "sasrec":
        backbone = SASRecBackbone(
            item_dim, data.max_len, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout
        )
    else:
        backbone = GPT2Backbone(args.gpt2_path, pretrained=not args.no_pretrained, freeze=args.freeze_backbone)
    return SeqRecommender(item_encoder, backbone, item_dim, data.valid_item_mask, temperature=args.temperature)


@torch.no_grad()
def evaluate(model, seq, target, history, item_pop, buckets, device, batch_size, topk=(10, 20)):
    """全量排序评测。返回整体指标和按目标电影训练热度分桶的指标。"""
    model.eval()
    max_k = max(topk)
    ranks = []
    for start in range(0, len(seq), batch_size):
        s = seq[start:start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            scores = model.predict(s).float()
        scores = scores.masked_fill(history[start:start + batch_size].to(device), float("-inf"))
        tgt = target[start:start + batch_size].to(device)
        tgt_score = scores.gather(1, tgt[:, None])
        # 悲观排名：分数不低于目标的电影数减 1（目标自身），同分算排在前面。
        ranks.append(((scores >= tgt_score).sum(1) - 1).cpu())
    ranks = torch.cat(ranks)

    def metrics(r):
        out = {}
        for k in topk:
            hit = (r < k).float()
            out[f"HR@{k}"] = hit.mean().item()
            out[f"NDCG@{k}"] = (hit / torch.log2(r.float() + 2)).mean().item()
        out["n_users"] = int(len(r))
        return out

    result = metrics(ranks)
    pop = item_pop[target]
    edges = [0] + list(buckets) + [float("inf")]
    result["buckets"] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pop >= lo) & (pop < hi)
        name = f"pop[{lo},{'inf' if hi == float('inf') else int(hi)})"
        result["buckets"][name] = metrics(ranks[sel]) if sel.any() else {"n_users": 0}
    return result


def case_study(model, data, user_row, device, k=10):
    seq = data.test_seq[user_row:user_row + 1].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        scores = model.predict(seq).float()[0]
    scores[data.test_history[user_row].to(device)] = float("-inf")
    top = scores.topk(k).indices.cpu().tolist()
    history = [i for i in data.test_seq[user_row].tolist() if i > 0][-5:]
    target = int(data.test_target[user_row])
    lines = [f"User {int(data.user_ids[user_row])} 最近观看："]
    lines += [f"    {i}: {data.title(i)}" for i in history]
    lines.append(f"  真实下一部：{target}: {data.title(target)}")
    lines.append(f"  Top-{k} 推荐：")
    lines += [f"    {'*' if i == target else ' '} {i}: {data.title(i)}" for i in top]
    return "\n".join(lines)


def main(args):
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    data = SeqData(args.data_path, max_len=args.max_len)
    print(data)
    model = build_model(args, data).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.tag}  trainable params: {n_params:,}  device: {device}")

    eval_kwargs = dict(item_pop=data.item_pop, buckets=args.buckets, device=device, batch_size=args.eval_batch_size)
    log = {"args": vars(args), "trainable_params": n_params, "epochs": []}
    start_time = time.time()
    best_state, best_score, best_epoch, wait = None, -math.inf, 0, 0

    if args.model != "popularity":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
        )
        n_train = len(data.train_seq)
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(n_train)
            total, count = 0.0, 0
            for start in tqdm(range(0, n_train, args.batch_size), desc=f"epoch {epoch}", leave=False, disable=not sys.stderr.isatty()):
                idx = perm[start:start + args.batch_size]
                seq = data.train_seq[idx].to(device)
                tgt = data.train_target[idx].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = model.loss(seq, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(idx)
                count += len(idx)
            train_loss = total / count
            valid = evaluate(model, data.valid_seq, data.valid_target, data.valid_history, **eval_kwargs)
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f}  valid HR@10={valid['HR@10']:.4f}  "
                f"NDCG@10={valid['NDCG@10']:.4f}  tau={model.log_tau.exp().item():.3f}"
                + (f"  peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if device == "cuda" else "")
            )
            log["epochs"].append({"epoch": epoch, "loss": train_loss, "HR@10": valid["HR@10"], "NDCG@10": valid["NDCG@10"]})
            if valid["NDCG@10"] > best_score:
                best_score, best_epoch, wait = valid["NDCG@10"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                    break
        model.load_state_dict(best_state)
        if args.save_ckpt:
            ckpt = HERE / "weight" / "ckpt" / f"{args.tag}_seed{args.seed}.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt)
            print(f"Saved checkpoint to {ckpt}")

    valid = evaluate(model, data.valid_seq, data.valid_target, data.valid_history, **eval_kwargs)
    test = evaluate(model, data.test_seq, data.test_target, data.test_history, **eval_kwargs)
    log.update({"best_epoch": best_epoch, "valid": valid, "test": test, "train_seconds": time.time() - start_time})
    print(f"Best epoch {best_epoch}  valid: HR@10={valid['HR@10']:.4f} NDCG@10={valid['NDCG@10']:.4f}")
    print(f"Test: HR@10={test['HR@10']:.4f} NDCG@10={test['NDCG@10']:.4f} HR@20={test['HR@20']:.4f} NDCG@20={test['NDCG@20']:.4f}")
    for name, m in test["buckets"].items():
        if m["n_users"]:
            print(f"  {name:>16}  n={m['n_users']:4d}  HR@10={m['HR@10']:.4f}  NDCG@10={m['NDCG@10']:.4f}")
    case = case_study(model, data, args.case_user_row, device)
    print(case)
    log["case_study"] = case

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="结果文件名前缀")
    p.add_argument("--model", choices=["seq", "popularity"], default="seq")
    p.add_argument("--backbone", choices=["sasrec", "gpt2"], default="gpt2")
    p.add_argument("--use_text", action="store_true")
    p.add_argument("--use_id", action="store_true")
    p.add_argument("--pooling", choices=["mean", "last"], default="mean")
    p.add_argument("--no_pretrained", action="store_true", help="GPT-2 随机初始化")
    p.add_argument("--freeze_backbone", action="store_true", help="冻结 GPT-2 用户模型")
    p.add_argument("--item_dim", type=int, default=768)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=None, help="默认：gpt2 5e-5，sasrec 1e-3")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buckets", type=int, nargs="+", default=[100, 500], help="冷启动分桶的热度阈值")
    p.add_argument("--case_user_row", type=int, default=0)
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_path", default=str(ROOT / "ml-1m"))
    p.add_argument("--gpt2_path", default=str(ROOT / "gpt2_model"))
    p.add_argument("--features", default=str(HERE / "weight" / "item_text_features.pt"))
    p.add_argument("--out_dir", default=str(HERE / "results"))
    args = p.parse_args()
    if args.lr is None:
        args.lr = 5e-5 if args.backbone == "gpt2" else 1e-3
    main(args)
