import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

class Muon(Optimizer):
    def __init__(self, params, lr=1e-4, momentum=0.95, weight_decay=0.0, ns_steps=5, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def zeropower_via_newtonschulz5(g, steps, eps):
        assert g.ndim >= 2
        original_shape = g.shape
        x = g.float().reshape(original_shape[0], -1)
        transposed = x.size(0) > x.size(1)
        if transposed:
            x = x.T

        norm = x.norm()
        if norm < eps:
            return torch.zeros_like(g)
        x = x / (norm + eps)

        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            xx_t = x @ x.T
            x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

        if transposed:
            x = x.T
        return x.reshape(original_shape).type_as(g)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim < 2:
                    raise ValueError("Muon should only receive parameters with at least 2 dimensions.")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = self.zeropower_via_newtonschulz5(buf, ns_steps, eps)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss


class MultiOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)


def set_optimizer_lrs(optimizer, adamw_lr, muon_lr=None):
    for param_group in optimizer.param_groups:
        group_type = param_group.get("group_type", "adamw")
        if group_type == "muon":
            param_group["lr"] = muon_lr if muon_lr is not None else adamw_lr
        else:
            param_group["lr"] = adamw_lr


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

# LLama改进之一，把LayerNorm改为RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# RoPE相关函数（位置编码）
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# 支持GQA，Llama3全使用GQA
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

# SwiGLU
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# class TransformerBlock(nn.Module):
#     def __init__(self, layer_id: int, args: ModelArgs):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.head_dim = args.dim // args.n_heads
#         self.attention = Attention(args)
#         self.feed_forward = FeedForward(
#             dim=args.dim,
#             hidden_dim=4 * args.dim,
#             multiple_of=args.multiple_of,
#             dropout=args.dropout,
#         )
#         self.layer_id = layer_id
#         self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

#     def forward(self, x, freqs_cos, freqs_sin):
#         h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
#         out = h + self.feed_forward.forward(self.ffn_norm(h))
#         return out

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # 【新增】：为每一层初始化一个专属的可学习伪查询向量 (Pseudo-query)
        # 维度与 hidden_dim 一致，使用正态分布初始化
        self.depth_query = nn.Parameter(torch.randn(args.dim) * (args.dim ** -0.5))

    # 【修改】：输入从单一的 x 变成了 history_states 列表
    def forward(self, history_states: list[torch.Tensor], freqs_cos, freqs_sin):
        # 1. 堆叠历史状态
        # history_states 是一个长度为 layer_id + 1 的列表，里面每个 tensor 形状为 (bsz, seqlen, dim)
        # stack 后的形状: (bsz, seqlen, num_past_layers, dim)
        stacked_history = torch.stack(history_states, dim=2)
        
        # 2. 计算跨层注意力得分
        # stacked_history: (bsz, seqlen, L, dim) 与 depth_query: (dim,) 做点积
        # 广播机制会自动处理前面维度，得到 scores 形状: (bsz, seqlen, L)
        scores = torch.matmul(stacked_history, self.depth_query) / math.sqrt(self.dim)
        
        # 3. 沿深度维度 (L) 做 Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. 加权聚合，得到当前层真正的输入 base_h
        # attn_weights 扩展为 (bsz, seqlen, L, 1)，然后与 history 相乘并沿 L 维度求和
        # 最终 base_h 形状恢复为: (bsz, seqlen, dim)
        base_h = (attn_weights.unsqueeze(-1) * stacked_history).sum(dim=2)
        
        # 5. 执行原本的 Block 内部逻辑（将原来的 x 全部替换为 base_h）
        # 注意：这里我们保留了 Block 内部的残差 (+)，使得整个 Block 的输出是一个完整的特征表达
        h = base_h + self.attention.forward(self.attention_norm(base_h), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     _bsz, seqlen = tokens.shape
    #     h = self.tok_embeddings(tokens)
    #     h = self.dropout(h)
    #     freqs_cos = self.freqs_cos[:seqlen]
    #     freqs_sin = self.freqs_sin[:seqlen]

    #     for layer in self.layers:
    #         h = layer(h, freqs_cos, freqs_sin)
    #     h = self.norm(h)

    #     if targets is not None:
    #         # if we are given some desired targets also calculate the loss
    #         logits = self.output(h)
    #         self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    #     else:
    #         # inference-time mini-optimization: only forward the output on the very last position
    #         logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
    #         self.last_loss = None

    #     return logits
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 【新增】：初始化历史状态池，最初只有 Embedding 的输出
        history_states = [h]

        for layer in self.layers:
            # 【修改】：将整个历史状态池传给当前层
            h = layer(history_states, freqs_cos, freqs_sin)
            
            # 【新增】：将当前层的输出追加到历史记录中，供更深层检索
            history_states.append(h)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            if return_all_logits:
                logits = self.output(h)
            else:
                logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        optimizer_type="adamw",
        muon_learning_rate=None,
    ):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if optimizer_type == "adamw":
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay, 'group_type': 'adamw_decay'},
                {'params': nodecay_params, 'weight_decay': 0.0, 'group_type': 'adamw_nodecay'},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed AdamW parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed AdamW parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
            return optimizer

        if optimizer_type != "muon":
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

        muon_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []
        for name, param in param_dict.items():
            is_muon_matrix = (
                param.dim() == 2
                and name.endswith(".weight")
                and not name.startswith("tok_embeddings")
                and not name.startswith("output")
            )
            if is_muon_matrix:
                muon_params.append(param)
            elif param.dim() >= 2:
                adamw_decay_params.append(param)
            else:
                adamw_nodecay_params.append(param)

        adamw_groups = [
            {'params': adamw_decay_params, 'weight_decay': weight_decay, 'group_type': 'adamw_decay'},
            {'params': adamw_nodecay_params, 'weight_decay': 0.0, 'group_type': 'adamw_nodecay'},
        ]
        adamw_optimizer = torch.optim.AdamW(adamw_groups, lr=learning_rate, betas=betas, **extra_args)

        muon_lr = learning_rate if muon_learning_rate is None else muon_learning_rate
        muon_optimizer = Muon(
            [{'params': muon_params, 'weight_decay': weight_decay, 'group_type': 'muon'}],
            lr=muon_lr,
            momentum=0.95,
            weight_decay=weight_decay,
            ns_steps=5,
        )

        num_muon_params = sum(p.numel() for p in muon_params)
        num_adamw_decay_params = sum(p.numel() for p in adamw_decay_params)
        num_adamw_nodecay_params = sum(p.numel() for p in adamw_nodecay_params)
        print(f"num Muon 2D matrix tensors: {len(muon_params)}, with {num_muon_params:,} parameters")
        print(f"num decayed AdamW fallback tensors: {len(adamw_decay_params)}, with {num_adamw_decay_params:,} parameters")
        print(f"num non-decayed AdamW fallback tensors: {len(adamw_nodecay_params)}, with {num_adamw_nodecay_params:,} parameters")
        print(f"using fused AdamW fallback: {use_fused}")

        return MultiOptimizer([adamw_optimizer, muon_optimizer])

    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = sum(p.numel() for p in self.parameters())
    #     cfg = self.params
    #     L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
    #     flops_per_token = 6*N + 12*L*H*Q*T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0/dt) # per second
    #     flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu

    #@torch.inference_mode()
    @torch.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next==eos:
                break

        return idx

    # def export(self, filepath='model.bin'):
    #     """export the model weights in fp32 into .bin file to be read from C"""
    #     f = open(filepath, 'wb')

    #     def serialize(t):
    #         d = t.detach().cpu().view(-1).numpy().astype(np.float32)
    #         b = struct.pack(f'{len(d)}f', *d)
    #         f.write(b)

    #     # first write out the header
    #     hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
    #     p = self.params
    #     n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    #     header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
    #                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    #     f.write(header)

    #     # next write out the embedding weights
    #     serialize(self.tok_embeddings.weight)

    #     # now all the layers
    #     # attention weights
    #     for layer in self.layers:
    #         serialize(layer.attention_norm.weight)
    #     for layer in self.layers:
    #         serialize(layer.attention.wq.weight)
    #     for layer in self.layers:
    #         serialize(layer.attention.wk.weight)
    #     for layer in self.layers:
    #         serialize(layer.attention.wv.weight)
    #     for layer in self.layers:
    #         serialize(layer.attention.wo.weight)
    #     # ffn weights
    #     for layer in self.layers:
    #         serialize(layer.ffn_norm.weight)
    #     for layer in self.layers:
    #         serialize(layer.feed_forward.w1.weight)
    #     for layer in self.layers:
    #         serialize(layer.feed_forward.w2.weight)
    #     for layer in self.layers:
    #         serialize(layer.feed_forward.w3.weight)
    #     # final rmsnorm
    #     serialize(self.norm.weight)
    #     # note: no need to write final classifier weights due to weight sharing
    #     # freqs_cis
    #     serialize(self.freqs_cos[:p.max_seq_len])
    #     serialize(self.freqs_sin[:p.max_seq_len])

    #     # write to binary file
    #     f.close()
    #     print(f"wrote {filepath}")