## 快速开始

### 1. 环境配置

项目依赖 MindSpore 2.x + Ascend NPU 运行环境（支持 PyNative 模式）。建议使用华为云 ModelArts 或本地已配置好 CANN 的昇腾设备。

### 2. 数据预处理

将金庸小说 `.txt` 文件转换为 MindRecord 格式：

```bash
python src/preprocess.py \
    --jinyong_dir ./data/jinyong \
    --tokenizer_path ./dataset/chinese_char_tokenizer.json \
    --output_file ./dataset/mindrecord \
    --file_partition 1
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--jinyong_dir` | 金庸小说 `.txt` 文件所在目录 |
| `--tokenizer_path` | ChineseCharTokenizer 路径（会自动训练并保存） |
| `--output_file` | 输出的 MindRecord 文件路径 |
| `--file_partition` | 分片数量 |

### 3. 训练

```bash
python train.py \
    --data_path ./dataset \
    --epoch 2 \
    --batch_size 12 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 500 \
    --lr_decay_iters 10000 \
    --sink_size 100
```

完整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `./dataset` | MindRecord 数据路径 |
| `--epoch` | 2 | 训练轮数 |
| `--batch_size` | 12 | 批次大小 |
| `--learning_rate` | 6e-5 | 最大学习率 |
| `--min_lr` | 6e-6 | 最小学习率（余弦退火） |
| `--warmup_iters` | 3000 | 线性热身步数 |
| `--lr_decay_iters` | 750000 | 衰减总步数 |
| `--weight-decay` | 0.1 | 权重衰减 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--beta1` / `--beta2` | 0.9 / 0.95 | Adam 超参数 |
| `--sink_size` | 100 | Sink 大小 |
| `--device_id` | 0 | Ascend 设备 ID |
| `--distribute` | false | 是否启用分布式训练 |

### 4. 生成

```bash
python generate.py \
    --ckpt_path ./GPT.ckpt \
    --tokenizer_path ./dataset/chinese_char_tokenizer.json
```

启动后交互式输入开头，模型会自动续写。内置生成超参数：

- 最大生成长度：200 tokens
- 采样温度（Temperature）：0.8（推荐 0.7–0.75）
- Top-K 采样：50（推荐 30–40）

