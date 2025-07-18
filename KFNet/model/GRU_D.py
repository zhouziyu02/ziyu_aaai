# -*- coding: utf-8 -*-
# GRU-D with SimplePatchGNN-style I/O
# Author: based on Wenjie Du et al. / adapted by ChatGPT-o3

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from einops import repeat


# -----------------------------------------------------------
# 1.  主网络
# -----------------------------------------------------------
class GRU_D(nn.Module):
    """
    GRU-D forecaster with the interface:
        ŷ = model(tp_pred, X, tp_true, mask=None)
            → (1, B, Lp, N)
    """

    def __init__(self, args):
        super().__init__()
        # ---------- 关键超参 ----------
        self.N          = args.ndim           # 变量数
        self.hist_len   = args.history        # 历史长度 L
        self.pred_len   = args.pred_len       # 预测长度 Lp
        self.hidden     = getattr(args, "rnn_hidden", 100)

        # ---------- 子模块 ----------
        n_steps   = self.hist_len + self.pred_len
        self.core = BackboneGRUD(
            n_steps=n_steps,
            n_features=self.N,
            rnn_hidden_size=self.hidden,
        )
        self.proj = nn.Linear(self.hidden, self.N)

    # ---------- 时间差 ----------
    @staticmethod
    def _to_delta(t):                         # (B,L) → (B,L,1)
        return F.pad(t[:, 1:] - t[:, :-1], (0, 0, 1, 0))

    # ---------- 掩码填充 ----------
    @staticmethod
    def _empirical_mean(x, m):                # both (B,L,N)
        valid = (1 - m)
        cnt   = valid.sum(1, keepdim=True)    # (B,1,N)
        cnt   = cnt.clamp_min(1)              # 避免除 0
        return (x * valid).sum(1, keepdim=True) / cnt

    @staticmethod
    def _locf(x, m):
        """last-observation-carried-forward"""
        # x, m : (B,L,N)
        B, L, N = x.shape
        locf = x.clone()
        for t in range(1, L):
            locf[:, t][m[:, t].bool()] = locf[:, t-1][m[:, t].bool()]
        return locf

    # ---------- forward ----------
    def forward(self, tp_pred, X, tp_true, mask=None):
        """
        tp_pred : (B,Lp)   future stamps
        X       : (B,L,N)  history
        tp_true : (B,L)    history stamps
        mask    : (B,L,N)  1 = observed, 0 = missing
        """
        B, L, N = X.shape
        Lp      = tp_pred.size(1)
        device  = X.device

        # --- 构造拼接序列 ---
        if mask is None:
            mask = torch.ones_like(X)

        # 未来占位
        X_pad     = torch.zeros(B, Lp, N, device=device)
        mask_pad  = torch.zeros_like(X_pad)   # 全缺失
        tp_concat = torch.cat([tp_true, tp_pred], dim=1)          # (B,L+Lp)
        X_all     = torch.cat([X, X_pad],     dim=1)              # (B,T,N)
        M_all     = torch.cat([mask, mask_pad], dim=1)            # (B,T,N)

        # --- 时间差、经验均值、LOCF 填充 ---
        deltas          = self._to_delta(tp_concat).unsqueeze(-1).repeat(1, 1, N)
        empirical_mean  = self._empirical_mean(X_all, M_all)      # (B,1,N)
        X_locf          = self._locf(X_all, M_all)

        # --- GRU-D 主干 ---
        reps, _ = self.core(X_all, M_all, deltas, empirical_mean, X_locf)
        recon   = self.proj(reps)                                 # (B,T,N)

        # --- 结果组装 ---
        y_hat   = recon[:, -Lp:]                                   # (B,Lp,N)
        return y_hat.unsqueeze(0).permute(0, 1, 2, 3)             # (1,B,Lp,N)


# -----------------------------------------------------------
# 2.  BackboneGRUD 与 TemporalDecay
#     （保持原论文实现，仅接口微调）
# -----------------------------------------------------------
class BackboneGRUD(nn.Module):
    def __init__(self, n_steps, n_features, rnn_hidden_size):
        super().__init__()
        self.n_steps        = n_steps
        self.n_features     = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.rnn_cell      = nn.GRUCell(n_features * 2 + rnn_hidden_size,
                                        rnn_hidden_size)
        self.decay_h       = TemporalDecay(n_features, rnn_hidden_size, diag=False)
        self.decay_x       = TemporalDecay(n_features, n_features, diag=True)

    def forward(self, X, M, Δ, μ, X_locf) -> Tuple[torch.Tensor, torch.Tensor]:
        B = X.size(0)
        h = torch.zeros(B, self.rnn_hidden_size, device=X.device)
        reps = []
        for t in range(self.n_steps):
            x_t, m_t, d_t = X[:, t], M[:, t], Δ[:, t]
            x_locf_t      = X_locf[:, t]

            γh = self.decay_h(d_t)
            γx = self.decay_x(d_t)
            h  = h * γh                               # 隐状态衰减
            reps.append(h)

            x_h  = γx * x_locf_t + (1 - γx) * μ.squeeze(1)
            x_in = m_t * x_t + (1 - m_t) * x_h        # 缺失值替换
            inp  = torch.cat([x_in, h, m_t], dim=1)
            h    = self.rnn_cell(inp, h)

        return torch.stack(reps, 1), h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if diag:
            eye = torch.eye(input_size)
            self.register_buffer("I", eye)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)

    def forward(self, Δ):
        out = F.linear(Δ, self.W * self.I if self.diag else self.W, self.b)
        return torch.exp(-F.relu(out))
