import numpy as np
import mindspore as ms
from mindspore import context, Tensor
import mindspore.nn as nn

# ==== 1. 选择运行后端 ====
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # 或改为 "GPU"

# ==== 2. 导入模型 ====
from model import GPT, GPTConfig, GPTWithLoss, GPTTrainOneStepCell

# ==== 3. 构建一个小模型，快速测试 ====
config = GPTConfig(
    block_size=32,   # 小序列
    vocab_size=8000, # 你的 tokenizer 词表
    n_layer=2,
    n_head=2,
    n_embd=64,
    dropout=0.0,
    bias=False
)
model = GPT(config)

# ==== 4. 伪造假数据（模拟 token ids） ====
batch_size = 4
seq_len = 32
fake_data = Tensor(
    np.random.randint(0, config.vocab_size, (batch_size, seq_len)),
    ms.int32
)

# ==== 5. 构建 Loss + Optimizer ====
gpt_loss = GPTWithLoss(model)

optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=1e-3,
    betas=(0.9, 0.95)
)

train_cell = nn.TrainOneStepCell(gpt_loss, optimizer)
train_cell.set_train()


# ==== 6. 跑几步训练，看看 loss 是否正常 ====
for step in range(5):
    loss = train_cell(fake_data)
    print(f"step {step} | loss = {loss.asnumpy()}")
