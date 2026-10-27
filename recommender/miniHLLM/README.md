# miniHLLM：用 GPT-2 做物品编码器和用户序列模型的序列推荐

把 HLLM（Hierarchical LLM for sequential recommendation）的两层结构缩到课程给的 GPT-2 small 上：

```
电影文本 "Title: Toy Story (1995). Genres: Animation, Children's, Comedy."
   │  Item LLM：冻结 GPT-2，掩码平均池化 → 768 维（item_features.py，离线一次）
   │  逐维标准化（去掉 GPT-2 隐藏状态的公共方向）→ 适配器 MLP + LayerNorm → 物品表 E (3953×768)
   ▼
用户历史 [i₁ … i_L] → 查表得向量序列 → User LLM：GPT-2 以 inputs_embeds 读入，因果注意力，全参数微调
   ▼
位置 t 的输出 h_t 与整张物品表做余弦相似度 / 温度 τ → 对全部电影 softmax 交叉熵，目标 i_{t+1}
```

模型里没有 ID embedding，物品的全部表示来自文本；打分侧和输入侧共用同一张物品表 E。

## 协议

- MovieLens-1M，每个用户按时间戳排序，leave-one-out：最后一条测试，倒数第二条验证，其余训练。
- 序列截到最近 100 条，训练时每个位置都预测下一部。
- 评测对全部 3883 部电影排序，屏蔽用户历史；平局按悲观处理。单目标下 NDCG@10 = 命中时 1/log2(rank+2)。
- 冷启动分桶：按目标电影在训练部分的出现次数分 <100、100–500、≥500 三桶。
- 最多 200 轮（SASRec 纯 ID 400 轮），验证 NDCG@10 连续 20 轮不涨早停，种子 42、2026、3407。

## 运行

```bash
cd recommender/miniHLLM
python item_features.py        # 提取 GPT-2 电影特征 → weight/item_text_features.pt
./run_all.sh                   # 主实验 6 个模型 × 3 种子 → results/main/
./run_extra.sh                 # 补充对照
python summarize.py            # 汇总表 → stdout（results/main/summary.md 为已生成版本）
python plot_results.py         # 三张图 → figures/
```

单次训练示例：

```bash
python train.py --tag hllm_text --backbone gpt2 --use_text --lr 1e-4 --max_len 100
python train.py --tag sasrec_id --backbone sasrec --use_id --item_dim 64 --dropout 0.1 --lr 1e-3
```

`train.py` 的开关：`--backbone sasrec|gpt2`、`--use_text`、`--use_id`、`--no_pretrained`（GPT-2 随机初始化）、`--freeze_backbone`（冻结 GPT-2 只训适配器）、`--pooling mean|last`。

## 结果（测试集，3 种子均值 ± 样本标准差）

| 模型 | 可训练参数 | 最佳轮次 | HR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0 | – | 0.037 | 0.018 |
| SASRec 纯 ID（400 轮） | 0.36M | 303, 246, 279 | 0.289 ± 0.006 | 0.164 ± 0.002 |
| SASRec 仅文本 | 0.16M | 115, 199, 192 | 0.292 ± 0.005 | 0.167 ± 0.002 |
| SASRec ID + 文本 | 0.42M | 146, 127, 101 | 0.297 ± 0.008 | 0.170 ± 0.005 |
| GPT-2 随机初始化，仅文本 | 87.6M | 16, 14, 16 | 0.283 ± 0.003 | 0.167 ± 0.003 |
| miniHLLM，冻结 GPT-2 | 1.8M | 54, 42, 53 | 0.283 ± 0.001 | 0.164 ± 0.002 |
| miniHLLM，文本 + ID | 90.7M | 24, 25, 27 | 0.292 ± 0.006 | 0.172 ± 0.003 |
| **miniHLLM，仅文本** | 87.6M | 24, 25, 25 | 0.294 ± 0.002 | **0.173 ± 0.001** |

冷启动分桶 NDCG@10：

| 模型 | <100 (507 人) | 100–500 (2195 人) | ≥500 (3338 人) |
| --- | ---: | ---: | ---: |
| SASRec 纯 ID | 0.052 ± 0.004 | 0.119 ± 0.005 | 0.210 ± 0.003 |
| SASRec 仅文本 | 0.042 ± 0.004 | 0.118 ± 0.004 | 0.218 ± 0.004 |
| SASRec ID + 文本 | 0.063 ± 0.005 | 0.123 ± 0.006 | 0.218 ± 0.006 |
| miniHLLM，仅文本 | 0.043 ± 0.003 | 0.119 ± 0.004 | **0.228 ± 0.005** |
| miniHLLM，冻结 GPT-2 | **0.065 ± 0.006** | **0.124 ± 0.004** | 0.206 ± 0.005 |

完整数字见 `results/main/summary.md`，每次运行的逐轮曲线、分桶指标和案例在同名 json 里。

## 结论

1. 不用任何 ID embedding，只靠 GPT-2 文本表示，miniHLLM 的整体 NDCG@10 达到所有模型最高，且三种子方差最小、25 轮收敛（SASRec 需要 100 到 300 轮）。
2. 语言预训练有效：同结构随机初始化低 0.6 个点 NDCG@10，差异远大于种子方差；随机初始化 15 轮即过拟合。
3. 加回 ID embedding 对 miniHLLM 没有帮助，物品文本表示已经足够。
4. 全参数微调的 GPT-2 把容量花在热门电影上：热门桶最好，冷门桶和纯 ID 一样。冻结 GPT-2 只训 1.8M 参数的适配器，冷门桶反而最好、热门桶最差，整体与收敛后的 SASRec 纯 ID 持平。微调程度是热门与冷门之间的权衡。
5. 文本特征在 SASRec 上的冷启动收益需要和 ID 一起用：仅文本的 SASRec 冷门桶并不比纯 ID 好。

## 设计过程中的两个坑

- GPT-2 隐藏状态各向异性很强，所有电影向量共享一个巨大的公共方向，任意两部电影的余弦都在 0.9 以上。对特征做逐维标准化后，GPT-2 版本 2 轮的验证 HR@10 从 0.04 升到 0.13。
- 打分用余弦相似度加可学习温度 τ，与 HLLM 的 InfoNCE 形式一致。评测分数强制 fp32 计算，bf16 在大数值下分辨率不够，会把大量电影判成同分；排名按悲观处理，同分算排在目标前面。

## 文件

| 文件 | 内容 |
| --- | --- |
| `item_features.py` | 冻结 GPT-2 提取电影文本向量（mean 和 last 两种池化） |
| `seq_data.py` | 时间排序、leave-one-out 划分、padding、历史掩码、热度统计 |
| `models.py` | ItemEncoder、SASRecBackbone、GPT2Backbone、SeqRecommender、PopularityRecommender |
| `train.py` | 训练、早停、全量排序评测、冷启动分桶、案例输出 |
| `run_all.sh` / `run_extra.sh` | 主实验与补充实验 |
| `summarize.py` / `plot_results.py` | 汇总表与图 |
| `weight/item_text_features.pt` | 电影文本特征 (3953×768)，两种池化 |
| `weight/ckpt/` | 种子 42 的最佳权重（不入库） |
| `results/main/` | 每次运行的日志与 json，`summary.md` |
| `results/sweep/`、`results/diag/` | 定超参用的短训练记录（学习率、序列长度） |
| `figures/` | 训练曲线、整体指标、冷启动分桶 |
