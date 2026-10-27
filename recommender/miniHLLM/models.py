"""miniHLLM 模型：Item 编码器 + User 序列骨干 + 共享物品表打分。

ItemEncoder   : 冻结 GPT-2 文本向量（可训练适配器）和 / 或 ID Embedding -> 整张物品表 E。
SASRecBackbone: 从零训练的因果 Transformer，作为对照。
GPT2Backbone  : 以 inputs_embeds 方式读入物品向量序列的 GPT-2（预训练或随机初始化）。
SeqRecommender: 位置 t 的输出与物品表 E 做余弦打分，对全部电影 softmax 交叉熵。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


class ItemEncoder(nn.Module):
    """返回整张物品表 E ∈ R^{n_items × dim}。文本分支与 ID 分支相加。"""

    def __init__(self, n_items, dim, text_features=None, use_text=True, use_id=False):
        super().__init__()
        if not (use_text or use_id):
            raise ValueError("use_text 和 use_id 至少开启一个")
        self.use_text, self.use_id = use_text, use_id
        if use_text:
            if text_features is None:
                raise ValueError("use_text=True 需要提供 text_features")
            # GPT-2 隐藏状态各向异性很强（所有向量共享一个大的公共方向），
            # 先按有效电影做逐维标准化，再进适配器。
            feats = text_features.float()
            valid = feats.abs().sum(1) > 0
            mean = feats[valid].mean(0, keepdim=True)
            std = feats[valid].std(0, keepdim=True) + 1e-6
            feats = torch.where(valid[:, None], (feats - mean) / std, torch.zeros_like(feats))
            self.register_buffer("text_features", feats)
            # 冻结 LLM 向量上的小适配器：替代 HLLM 中对 Item LLM 的微调。
            self.adapter = nn.Sequential(
                nn.Linear(text_features.shape[1], dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
        if use_id:
            self.id_embedding = nn.Embedding(n_items, dim, padding_idx=0)
            nn.init.normal_(self.id_embedding.weight, std=0.02)

    def forward(self):
        table = 0
        if self.use_text:
            table = table + self.adapter(self.text_features)
        if self.use_id:
            table = table + self.id_embedding.weight
        return table


class SASRecBackbone(nn.Module):
    def __init__(self, dim, max_len, n_layers=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            dim, n_heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dim

    def forward(self, x, pad_mask):
        # x: [B, L, dim]; pad_mask: [B, L]，True 表示真实位置。
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        x = self.dropout(x + self.pos_embedding(pos)[None])
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.encoder(x, mask=causal, src_key_padding_mask=~pad_mask)
        return self.norm(h)


class GPT2Backbone(nn.Module):
    def __init__(self, model_path, pretrained=True, freeze=False):
        super().__init__()
        if pretrained:
            self.gpt2 = GPT2Model.from_pretrained(model_path, local_files_only=True)
        else:
            self.gpt2 = GPT2Model(GPT2Config.from_pretrained(model_path, local_files_only=True))
        # 词表 Embedding 不参与 inputs_embeds 路径，删掉省显存。
        self.gpt2.wte = nn.Identity()
        if freeze:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)
        self.out_dim = self.gpt2.config.n_embd

    def forward(self, x, pad_mask):
        return self.gpt2(inputs_embeds=x, attention_mask=pad_mask.long()).last_hidden_state


class SeqRecommender(nn.Module):
    def __init__(self, item_encoder, backbone, item_dim, valid_item_mask, temperature=0.05, scoring="cosine"):
        super().__init__()
        self.scoring = scoring
        self.item_encoder = item_encoder
        self.backbone = backbone
        hidden = backbone.out_dim
        self.in_proj = nn.Linear(item_dim, hidden) if item_dim != hidden else nn.Identity()
        self.out_proj = nn.Linear(hidden, item_dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))
        self.register_buffer("valid_item_mask", valid_item_mask.clone())

    def encode(self, seq):
        """seq: [B, L] 电影编号，0 为填充。返回位置输出 [B, L, d] 和物品表 [N, d]。"""
        table = self.item_encoder()
        pad_mask = seq > 0
        x = self.in_proj(table[seq])
        h = self.backbone(x, pad_mask)
        return self.out_proj(h), table

    def scores(self, h, table):
        """余弦相似度 / 温度，或纯点积；无效电影置为 -inf。h: [..., d] -> [..., N]。"""
        with torch.autocast("cuda", enabled=False):
            return self._scores(h.float(), table.float())

    def _scores(self, h, table):
        if self.scoring == "cosine":
            tau = self.log_tau.clamp(math.log(0.01), math.log(1.0)).exp()
            logits = F.normalize(h, dim=-1) @ F.normalize(table, dim=-1).t() / tau
        else:
            logits = h @ table.t()
        return logits.masked_fill(~self.valid_item_mask, float("-inf"))

    def loss(self, seq, target):
        h, table = self.encode(seq)
        logits = self.scores(h, table)                      # [B, L, N]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100
        )

    @torch.no_grad()
    def predict(self, seq):
        """取每条序列最后一个真实位置的输出，返回 [B, N] 分数。"""
        h, table = self.encode(seq)
        last = (seq > 0).sum(1) - 1
        h_last = h[torch.arange(seq.size(0), device=seq.device), last]
        return self.scores(h_last, table)


class PopularityRecommender(nn.Module):
    """按训练部分的交互次数排序，所有用户得到同一张榜单（再屏蔽各自历史）。"""

    def __init__(self, item_pop, valid_item_mask):
        super().__init__()
        self.register_buffer("item_pop", item_pop.float())
        self.register_buffer("valid_item_mask", valid_item_mask.clone())

    @torch.no_grad()
    def predict(self, seq):
        scores = self.item_pop[None].expand(seq.size(0), -1).clone()
        return scores.masked_fill(~self.valid_item_mask, float("-inf"))
