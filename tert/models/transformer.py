"""Causal transformer over interleaved (obs, action) tokens.

The action head reads observation-token positions, so a_t depends on o_1..o_t
and nothing later. No returns-to-go: there is no return to condition on when the
target is the teacher's action.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    obs_dim: int = 48
    act_dim: int = 12
    context_len: int = 20
    n_blocks: int = 3
    embed_dim: int = 256
    n_heads: int = 1
    dropout: float = 0.05
    max_timestep: int = 4096
    # Post-LN matches the original; pre-LN is the modern default and trains more
    # stably at depth. Switchable so the choice stays an experiment, not an accident.
    pre_ln: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.proj_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, return_weights=False):
        B, L, C = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        if not return_weights:
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
            weights = None
        else:
            # Explicit path so attention maps can be inspected.
            scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            weights = scores.softmax(dim=-1)
            attn = F.dropout(weights, self.dropout, self.training) @ v

        attn = attn.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(attn)), weights


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.pre_ln = cfg.pre_ln
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x, return_weights=False):
        if self.pre_ln:
            a, w = self.attn(self.ln1(x), return_weights)
            x = x + a
            x = x + self.mlp(self.ln2(x))
        else:
            a, w = self.attn(x, return_weights)
            x = self.ln1(x + a)
            x = self.ln2(x + self.mlp(x))
        return x, w


class TerrainTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_obs = nn.Linear(cfg.obs_dim, cfg.embed_dim)
        self.embed_act = nn.Linear(cfg.act_dim, cfg.embed_dim)
        self.embed_time = nn.Embedding(cfg.max_timestep, cfg.embed_dim)
        self.embed_ln = nn.LayerNorm(cfg.embed_dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.predict_action = nn.Linear(cfg.embed_dim, cfg.act_dim)

    def forward(self, timesteps, obs, actions, return_weights=False):
        """obs (B,T,obs_dim), actions (B,T,act_dim), timesteps (B,T) -> (B,T,act_dim)."""
        B, T, _ = obs.shape
        t = self.embed_time(timesteps)
        tokens = torch.stack([self.embed_obs(obs) + t, self.embed_act(actions) + t], dim=2)
        h = self.embed_ln(tokens.reshape(B, 2 * T, self.cfg.embed_dim))

        weights = []
        for block in self.blocks:
            h, w = block(h, return_weights)
            if w is not None:
                weights.append(w)

        h = h.reshape(B, T, 2, self.cfg.embed_dim)
        actions_pred = self.predict_action(h[:, :, 0])  # observation-token positions
        return (actions_pred, weights) if return_weights else actions_pred

    @torch.no_grad()
    def act(self, timesteps, obs, actions):
        """Greedy inference on the trailing window; returns the action for the last step."""
        return self.forward(timesteps, obs, actions)[:, -1]
