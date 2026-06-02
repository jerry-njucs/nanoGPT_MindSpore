from dataclasses import dataclass
from typing import Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
from mindspore import ops
from mindspore.common.initializer import Normal, initializer
from mindspore.ops import functional as F

# TODO: 实现 GPT 模型结构
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 5440
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster # fmt:skip


class LayerNorm(nn.Cell):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = ms.Parameter(ms.ops.ones((ndim,), ms.float32))  # 缩放系数
        if bias:
            self.bias_param = ms.Parameter(ms.ops.zeros((ndim,), ms.float32))
        else:
            self.bias_param = None

    def construct(self, x):
        eps = 1e-5
        mean = x.mean(axis=-1, keep_dims=True)
        variance = ops.sqrt(((x - mean) ** 2).mean(axis=-1, keep_dims=True) + eps)

        norm_x = (x - mean) / variance
        if self.bias_param is not None:
            return norm_x * self.weight + self.bias_param
        else:
            return norm_x * self.weight


class CausalSelfAttention(nn.Cell):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Dense(config.n_embd, 3 * config.n_embd, has_bias=config.bias)
        self.proj = nn.Dense(config.n_embd, config.n_embd, has_bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)

        # kv cache
        self.use_kvcache = False
        self.k_cache = None
        self.v_cache = None

        # 因果掩码
        mask = np.triu(np.ones((config.block_size, config.block_size), dtype=np.float32), k=1)
        mask = mask * (-1e9)
        self.mask = ms.Tensor(mask, ms.float32)

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def get_cache_len(self):
        """返回当前缓存的序列长度"""
        if self.k_cache is None:
            return 0
        return self.k_cache.shape[2]

    def construct(self, x):
        B, T, C = x.shape  # T 是当前输入的长度（使用 KV Cache 时可能是 1）

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
        q, k, v = ops.transpose(q, (0, 2, 1, 3)), ops.transpose(k, (0, 2, 1, 3)), ops.transpose(v, (0, 2, 1, 3))

        if self.use_kvcache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = ops.concat([self.k_cache, k], axis=2)
                self.v_cache = ops.concat([self.v_cache, v], axis=2)
            k = self.k_cache
            v = self.v_cache
            # 注意：这里不更新 T，因为 T 是 query 的长度，用于最后 reshape

        scale = 1.0 / ops.sqrt(ms.Tensor(self.head_dim, ms.float32))
        # attention: (B, nh, T_q, head_dim) @ (B, nh, head_dim, T_kv) -> (B, nh, T_q, T_kv)
        attention = ops.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # 因果掩码：只在非 KV Cache 模式或 prefill 阶段使用
        if not self.use_kvcache:
            mask = self.mask[:T, :T]
            mask = ops.expand_dims(ops.expand_dims(mask, 0), 0)
            attention = attention + mask
        # 使用 KV Cache 且 T=1 时（decode 阶段），不需要掩码

        attention = ops.softmax(attention, axis=-1)
        attention = self.dropout(attention)

        # y: (B, nh, T_q, T_kv) @ (B, nh, T_kv, head_dim) -> (B, nh, T_q, head_dim)
        y = ops.matmul(attention, v)
        # reshape 使用 T（query 的长度），不是缓存长度
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        y = self.proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Cell):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.SequentialCell([  # 使用SequentialCell来组合多个层
            nn.Dense(config.n_embd, 4 * config.n_embd, has_bias=config.bias),
            nn.GELU(),
            nn.Dense(4 * config.n_embd, config.n_embd, has_bias=config.bias),
            nn.Dropout(p=config.dropout),
        ])

    def construct(self, x):
        return self.fc(x)

class Block(nn.Cell):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def construct(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Cell):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # transformer blocks
        self.blocks = nn.SequentialCell([Block(config) for _ in range(config.n_layer)])
        # layer norm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        # output head
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)
        # init weights
        self._init_all_weights()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.get_parameters())
        if non_embedding:
            n_params -= self.wpe.embedding_table.numel()
        return n_params
    
    def _init_all_weights(self):
        """遍历所有子模块并初始化权重"""
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer(Normal(mean=0.0, sigma=0.02), cell.weight.shape)
                )
                if cell.has_bias and cell.bias is not None:
                    cell.bias.set_data(
                        ms.ops.zeros(cell.bias.shape, ms.float32)
                    )
            elif isinstance(cell, nn.Embedding):
                cell.embedding_table.set_data(
                    initializer(Normal(mean=0.0, sigma=0.02), cell.embedding_table.shape)
                )
            elif isinstance(cell, LayerNorm):
                cell.weight.set_data(
                    ms.ops.ones(cell.weight.shape, ms.float32)
                )
                if cell.bias_param is not None:
                    cell.bias_param.set_data(
                        ms.ops.zeros(cell.bias_param.shape, ms.float32)
                    )

    """
    def _init_weights(self, module):
        if isinstance(module, ms.Parameter):
            module.set_data(initializer(Normal(mean=0.0, sigma=0.02), module.shape))
        elif isinstance(module, nn.Linear):
            module.weight.set_data(
                initializer(Normal(mean=0.0, sigma=0.02), module.weight.shape)
            )
            if module.bias is not None:
                module.bias.set_data(
                    initializer(Normal(mean=0.0, sigma=0.02), module.bias.shape)
                )
        elif isinstance(module, nn.Embedding):
            embedding_table = getattr(module, "embedding_table", None)
            if embedding_table is not None:
                embedding_table.set_data(
                    initializer(Normal(mean=0.0, sigma=0.02), embedding_table.shape)
                )
    """
    def construct(self, idx: ms.Tensor, targets: Optional[ms.Tensor] = None, start_pos: int = 0):
        B, T = idx.shape

        token_embeddings = self.wte(idx)  # (B, T, C)

        # 位置编码：考虑 start_pos 偏移（用于 KV Cache）
        pos_ids = ops.arange(start_pos, start_pos + T)
        position = self.wpe(pos_ids)  # (T, C)

        x = token_embeddings + position  # (B, T, C)，自动广播
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                ops.reshape(logits, (-1, logits.shape[-1])),
                ops.reshape(targets, (-1,)),
            )

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        def decay_filter(x: ms.Parameter) -> bool:
            return len(x.shape) >= 2

        params = self.trainable_params()
        decay_params = list(filter(decay_filter, params))
        other_params = list(filter(lambda x: not decay_filter(x), params))
        group_params = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": other_params, "weight_decay": 0.0},
            {"order_params": params},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in other_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with"
            f" {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(other_params)}, with"
            f" {num_nodecay_params:,} parameters"
        )
        optimizer = nn.AdamWeightDecay(
            group_params,
            learning_rate=learning_rate,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=0.0,
        )

        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成，使用 KV Cache 加速
        """
        self.set_train(False)

        # 启用 KV Cache 并重置
        for block in self.blocks:
            block.attn.use_kvcache = True
            block.attn.reset_kv_cache()

        # Prefill 阶段：处理整个 prompt
        idx_cond = idx[:, -self.config.block_size:]
        logits, _ = self.construct(idx_cond, start_pos=0)
        
        for _ in range(max_new_tokens):
            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                kth = ops.top_k(next_logits, top_k)[0][:, -1:]
                next_logits = ops.where(next_logits < kth, -1e10, next_logits)

            probs = ops.softmax(next_logits, axis=-1)
            next_idx = ops.multinomial(probs, num_samples=1)

            idx = ops.concat([idx, next_idx], axis=1)

            # Decode 阶段：只传入新生成的 token
            cur_pos = idx.shape[1] - 1  # 当前位置（用于位置编码）
            if cur_pos < self.config.block_size:
                logits, _ = self.construct(next_idx, start_pos=cur_pos)

        # 关闭 KV Cache
        for block in self.blocks:
            block.attn.use_kvcache = False
            block.attn.reset_kv_cache()

        return idx


class GPTWithLoss(nn.Cell):
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def construct(self, input_ids: ms.Tensor):
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        _, loss = self.gpt(tokens, labels)
        return loss


_grad_scale = ops.composite.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.reciprocal(scale)


class GPTTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    通用 GPT 训练封装：
    1. 包含混合精度 (FP16) 的 Loss Scaling
    2. 包含梯度裁剪 (Gradient Clipping) 防止 Loss NaN
    """

    def __init__(self, network, optimizer, scale_sense, gradient_clip=1.0):
        super(GPTTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.gradient_clip = gradient_clip
        self.hyper_map = ops.HyperMap()

    def construct(self, input_ids: ms.Tensor):
        weights = self.weights
        loss = self.network(input_ids)

        status, scaling_sens = self.start_overflow_check(loss, self.scale_sense)
        grads = self.grad(self.network, weights)(input_ids, scaling_sens)

        # 2. 梯度还原（Unscale）：把梯度除以 scale，变回真实大小
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)

        grads = ops.clip_by_global_norm(grads, self.gradient_clip)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        if not overflow:
            self.optimizer(grads)
            return loss
        else:
            print("Gradient overflow detected, skipping step and reducing loss scale.")
            return ops.zeros_like(loss)