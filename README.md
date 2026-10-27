# 基于大语言模型增强的推荐系统

上海交通大学课程项目. 在 MovieLens-1M 上实现两个用 GPT-2 增强的电影推荐系统:

- **第一部分 `recommender/base_LLM_NCF/`**: 课程基础框架. 冻结的 GPT-2 把电影标题与类型编码为文本向量, 与用户, 电影 ID 向量拼接后由 NCF 打分, 用 BPR 损失训练.
- **第二部分 `recommender/miniHLLM/`**: 参照 ByteDance HLLM, 用 GPT-2 同时作为物品编码器与用户序列模型的序列推荐模型, 不含 ID embedding. 包含 SASRec 等对照模型与按电影热度分桶的评测.

方法与结果详见 [`report/report.pdf`](report/report.pdf).

## 目录结构

```text
.
├── README.md
├── PROJECT_DESC.md              # 课程提供的第一部分任务说明
├── requirements.txt
├── ml-1m/                       # MovieLens-1M 数据集 (Git LFS)
│   ├── movies.dat
│   ├── ratings.dat
│   ├── users.dat
│   └── README
├── gpt2_model/                  # GPT-2 small 预训练权重与分词器 (Git LFS)
├── recommender/
│   ├── base_LLM_NCF/            # 第一部分: GPT-2 特征增强的 NCF
│   │   ├── gpt2.py              # 提取电影文本特征 -> weight/movie_features.pt
│   │   ├── dataset.py           # 数据读取与 8:1:1 随机划分
│   │   ├── dataloader.py        # 负采样与评测数据组织
│   │   ├── recommender.py       # GPT-2 + NCF 模型与 BPR 损失
│   │   ├── trainer.py           # 训练与评测 (课程提供)
│   │   ├── main.py              # 运行入口
│   │   └── weight/              # 文本特征与训练好的 checkpoint
│   └── miniHLLM/                # 第二部分: miniHLLM 序列推荐
│       ├── item_features.py     # 提取电影文本特征 -> weight/item_text_features.pt
│       ├── seq_data.py          # 按时间排序, leave-one-out 划分, 填充, 热度统计
│       ├── models.py            # ItemEncoder, SASRec / GPT-2 骨干, 推荐模型, Popularity
│       ├── train.py             # 训练, 早停, 全量排序评测, 热度分桶, 推荐案例
│       ├── run_all.sh           # 主实验
│       ├── run_extra.sh         # 补充实验
│       ├── sweep.sh, diag.sh    # 选择学习率与序列长度的预实验
│       ├── summarize.py         # 汇总结果表
│       ├── plot_results.py      # 绘图 -> figures/
│       ├── weight/              # 电影文本特征; ckpt/ 为模型权重 (不入库)
│       ├── results/             # main/ 正式实验, sweep/ 与 diag/ 预实验的日志与 json
│       └── figures/             # 训练曲线, 整体指标, 热度分桶图
└── report/
    ├── report.pdf               # 课程报告
    └── source/                  # 报告 LaTeX 源码与插图
```

## 运行环境

实验所用环境:

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Linux |
| GPU | NVIDIA RTX 3090 (24 GB) |
| Python | 3.12 |
| PyTorch | 2.2.1 + CUDA 12.1 |
| Transformers | 4.40.2 |
| 其他 | numpy 1.26.4, pandas 2.2.2, tqdm 4.66.5, matplotlib (仅绘图需要) |

第一部分在 CPU 上也能运行. 第二部分的 GPT-2 系列模型需要显存 24 GB 的 GPU; SASRec 系列只需约 1.2 GB 显存.

安装:

```bash
conda create -n hybridrec python=3.12 -y
conda activate hybridrec
# 按本机 CUDA 版本安装 PyTorch, 见 https://pytorch.org/get-started/previous-versions/
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install matplotlib            # 仅 plot_results.py 需要
```

## 数据与预训练模型

数据集和 GPT-2 权重通过 Git LFS 存放在仓库中, 克隆后拉取即可:

```bash
git lfs install
git clone https://github.com/xiuyunMarx/SJTU-course-project-hybrid-rec.git
cd SJTU-course-project-hybrid-rec
git lfs pull
```

Git LFS 安装方法见 <https://git-lfs.com>.

如 LFS 下载失败, 也可以从官方来源下载, 放到仓库根目录下的对应位置:

| 资源 | 下载链接 | 存放位置 |
| --- | --- | --- |
| MovieLens-1M | <https://files.grouplens.org/datasets/movielens/ml-1m.zip> | 解压到 `ml-1m/` |
| GPT-2 small | <https://huggingface.co/openai-community/gpt2> | `gpt2_model/` |
| GPT-2 small (国内镜像) | <https://hf-mirror.com/openai-community/gpt2> | `gpt2_model/` |

```bash
# MovieLens-1M
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip                   # 得到 ml-1m/

# GPT-2 small; 国内网络可先执行 export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openai-community/gpt2 \
    config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json \
    --local-dir gpt2_model
```

电影文本特征 (`weight/*.pt`) 和模型权重可以由下面的脚本重新生成.

## 运行方法

所有脚本按自身所在位置定位仓库根目录, 默认读取根目录下的 `ml-1m/` 与 `gpt2_model/`.

### 第一部分: GPT-2 特征增强的 NCF

```bash
cd recommender/base_LLM_NCF
python gpt2.py                    # 提取电影文本特征, 形状 (3953, 768)
python main.py --quick --epochs 2 # 快速检查, 只用前 500 位用户
python main.py --epochs 3         # 全量训练
```

训练过程中输出 loss, HR@10 与 NDCG@10; 结束时输出验证集与测试集指标, 并为一名用户生成 Top-10 推荐. checkpoint 保存在 `weight/` 下.

### 第二部分: miniHLLM

```bash
cd recommender/miniHLLM
python item_features.py           # 提取电影文本特征 -> weight/item_text_features.pt
./run_all.sh                      # 主实验: 6 个模型 x 3 个种子 -> results/main/
./run_extra.sh                    # 补充实验: SASRec 400 轮, 仅文本 SASRec, 冻结 GPT-2 等
python summarize.py               # 汇总表
python plot_results.py            # 绘图 -> figures/
```

`run_all.sh` 与 `run_extra.sh` 会跳过 `results/main/` 中已有结果的运行. 在 RTX 3090 上, GPT-2 系列每次训练约 15 分钟, SASRec 系列约 3.5 分钟.

单次训练示例:

```bash
# miniHLLM (仅文本, 预训练 GPT-2 全参数微调)
python train.py --tag hllm_text --backbone gpt2 --use_text --lr 1e-4 --max_len 100 --epochs 200 --patience 20
# SASRec (仅 ID)
python train.py --tag sasrec_id --backbone sasrec --use_id --item_dim 64 --dropout 0.1 --lr 1e-3 --max_len 100 --epochs 400 --patience 20
```

`train.py` 的主要开关:

| 参数 | 作用 |
| --- | --- |
| `--backbone sasrec\|gpt2` | 用户序列模型 |
| `--use_text` / `--use_id` | 电影表示使用文本向量 / ID 向量, 可同时开启 |
| `--no_pretrained` | GPT-2 随机初始化 |
| `--freeze_backbone` | 冻结 GPT-2, 只训练适配器与投影层 |
| `--model popularity` | 热门推荐基线 |
| `--save_ckpt` | 保存最佳权重到 `weight/ckpt/` |

每次运行在 `results/main/` 下生成同名的 `.log` 与 `.json`, json 中包含逐轮验证曲线, 测试集指标, 热度分桶指标和推荐案例.
