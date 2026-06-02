# nanoGPT-MindSpore：金庸风格文本续写

基于 **MindSpore** 框架实现 GPT-2 Small（124M 参数量）模型，在金庸 15 部小说全集上训练，实现中文武侠风格的文本续写。

## 项目概览

将一个 **GPT-2 Small**（124M 参数）从 PyTorch 迁移到 **MindSpore + Ascend NPU** 平台，并用金庸全集中文语料进行训练。

---

## 模型架构

| 配置项 | 数值 | 说明 |
|-------|------|------|
| 模型类型 | GPT-2 Small | 纯 Transformer Decoder |
| 层数 (`n_layer`) | 12 | — |
| 注意力头数 (`n_head`) | 12 | — |
| 隐藏层维度 (`n_embd`) | 768 | — |
| 最大上下文 (`block_size`) | 1024 | 训练时实际使用 256 |
| 词表大小 (`vocab_size`) | 5440 | 原始字表 5424，对齐到 64 倍数 |
| Dropout | 0.0 | 关闭 |
| 参数量 | **~124M** | — |

### 关键组件

- **字符级 Tokenizer**（`ChineseCharTokenizer`）：基于字符频率统计，选取 5420 个最常用汉字 + 4 个特殊 token（PAD / UNK / BOS / EOS），非 BPE/Subword。
- **Causal Self-Attention**：带因果掩码的多头自注意力。
- **KV Cache**：生成时缓存 K/V 矩阵，Prefill-Decode 两阶段加速，避免重复计算已生成 token。
- **Weight-untied** 输出头：`lm_head` 与 `wte` 不共享权重。
- **FP16 混合精度训练**：动态 Loss Scaling（初始 1024，scale_factor=2，scale_window=1000）。
- **Gradient Clipping**：梯度全局裁剪，阈值 1.0。
- **AdamW 优化器**：仅对 `ndim ≥ 2` 的参数施加 weight decay。

---

## 项目结构

```
├── train.py                          # 训练入口
├── generate.py                       # 交互式生成入口
├── src/
│   ├── model.py                      # GPT 模型架构（核心）
│   ├── chinese_char_tokenizer.py     # 中文字符级 Tokenizer
│   ├── dataset.py                    # MindRecord 数据集加载
│   ├── preprocess.py                 # 数据预处理（.txt → MindRecord）
│   ├── utils.py                      # 学习率调度、文件读取等工具
│   └── local_test.py                 # CPU 本地冒烟测试
├── data/
│   └── jinyong/                      # 15 部金庸小说 .txt 原文
│       ├── 射雕英雄传.txt
│       ├── 天龙八部.txt
│       ├── 鹿鼎记.txt
│       └── ...
├── dataset/
│   ├── chinese_char_tokenizer.json   # 预训练 Tokenizer（5424 字表）
│   └── mindrecord /                  # 预处理后的 MindRecord 数据
├── Exp_Report/
│   └── Report.md                     # 实验报告与生成样例
└── *.ckpt                            # 预训练检查点
```

