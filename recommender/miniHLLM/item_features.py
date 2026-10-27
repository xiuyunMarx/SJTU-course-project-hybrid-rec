"""miniHLLM 的 Item LLM：用冻结的 GPT-2 把每部电影的文本编码成向量，离线保存。

同时保存两种读出方式，供训练时选择：
- mean：掩码平均池化；
- last：最后一个真实 token 的隐藏状态（对应 HLLM 中附加 [ITEM] 读出 token 的做法）。
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2Model, GPT2Tokenizer

HERE = Path(__file__).resolve().parent          # recommender/miniHLLM
ROOT = HERE.parents[1]                          # 仓库根目录


def build_texts(movies):
    return [
        f"Title: {row.title}. Genres: {row.genres.replace('|', ', ')}."
        for row in movies.itertuples()
    ]


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    movies = pd.read_csv(
        Path(args.data_path) / "movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    texts = build_texts(movies)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = GPT2Model.from_pretrained(args.model_path, local_files_only=True)
    model = model.to(device).eval()

    mean_chunks, last_chunks = [], []
    for start in range(0, len(texts), args.batch_size):
        enc = tokenizer(
            texts[start:start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**enc).last_hidden_state            # [B, L, 768]
        attn = enc["attention_mask"]                        # [B, L]
        mask = attn.unsqueeze(-1).to(hidden.dtype)
        mean_chunks.append(((hidden * mask).sum(1) / mask.sum(1)).cpu())
        last_idx = attn.sum(1) - 1
        last_chunks.append(hidden[torch.arange(hidden.size(0), device=device), last_idx].cpu())

    mean = torch.cat(mean_chunks)
    last = torch.cat(last_chunks)
    n_items = int(movies["movie_id"].max()) + 1
    ids = torch.as_tensor(movies["movie_id"].to_numpy(), dtype=torch.long)

    out = {"mean": torch.zeros(n_items, mean.shape[1]), "last": torch.zeros(n_items, last.shape[1])}
    out["mean"][ids] = mean
    out["last"][ids] = last
    out["movie_ids"] = ids

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f"Device: {device}")
    print(f"Encoded {len(texts)} movies -> {output}")
    print(f"Feature table shape: {tuple(out['mean'].shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 miniHLLM 的电影文本特征")
    parser.add_argument("--data_path", default=str(ROOT / "ml-1m"))
    parser.add_argument("--model_path", default=str(ROOT / "gpt2_model"))
    parser.add_argument("--output", default=str(HERE / "weight" / "item_text_features.pt"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=32)
    main(parser.parse_args())
