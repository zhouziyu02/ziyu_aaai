import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# # -----------------------------------------------------------
# #   1. Patch 内表征：Temporal-Kernel Mixture (TTKMN)
# # -----------------------------------------------------------
# class TTKMN(nn.Module):
#     """
#     Temporal-Kernel Mixture Network
#     • K 个高斯核：可学习中心 c 和宽度 alpha
#     • 输出：K 维加权表示 + 有/无观测 flag  ==>  (B, K+1)
#     """
#     def __init__(self, K: int):
#         super().__init__()
#         self.K = K
#         # learnable centres   c_k ∈ [0,1]
#         self.c = nn.Parameter(torch.linspace(0, 1, K))
#         # learnable log-scale
#         self.log_alpha = nn.Parameter(torch.zeros(K))
#         # optional gated weights per kernel
#         self.gate = nn.Parameter(torch.zeros(K))          # (K,)

#     def forward(self, t, x, mask):
#         """
#         t, x, mask : shape (B, L, 1)
#         """
#         alpha = self.log_alpha.exp() + 1e-6               # (K,)
#         td    = t - self.c.view(1, 1, self.K)             # (B,L,K)
#         w     = torch.exp(-0.5 * (td ** 2) / alpha.view(1, 1, self.K) ** 2)
#         w     = w * mask                                  # 遮掉缺失
#         a     = w / (w.sum(1, keepdim=True) + 1e-8)       # (B,L,K) 归一化

#         h     = torch.einsum('blk,bld->bk', a, x)         # (B,K)
#         h     = h * torch.sigmoid(self.gate)              # 门控

#         flag  = (mask.sum(1) > 0).float()                 # (B,1)
#         return torch.cat([h, flag], -1)                   # (B, K+1)


# # -----------------------------------------------------------
# #   2. 位置编码 (PositionalEncoding)
# # -----------------------------------------------------------
# class PositionalEncoding(nn.Module):
#     def __init__(self, d, max_len=512):
#         super().__init__()
#         pos = torch.arange(0, max_len).unsqueeze(1)
#         div = torch.exp(torch.arange(0, d, 2)*(-math.log(10000.0)/d))
#         pe  = torch.zeros(max_len, d)
#         pe[:, 0::2] = torch.sin(pos*div) 
#         pe[:, 1::2] = torch.cos(pos*div)
#         self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d)

#     def forward(self, x):  # x: (B, L, d)
#         return x + self.pe[:, :x.size(1)]


# # -----------------------------------------------------------
# #   3. 随机特征映射辅助函数 (与原代码一致)
# # -----------------------------------------------------------
# def random_fourier_features(x, W, b):
#     proj = torch.einsum('bhmd,hdR->bhmR', x, W)  # (B, h, M, r//2)
#     proj = proj + b.unsqueeze(1).unsqueeze(2)   # (B, h, M, r//2)
#     cos_part = torch.cos(proj)
#     sin_part = torch.sin(proj)
#     phi = torch.cat([cos_part, sin_part], dim=-1)  
#     phi = phi / math.sqrt(phi.size(-1)//2)  
#     return phi


# # -----------------------------------------------------------
# #   4. 线性注意力模块 (基于随机特征映射 + 频域)
# # -----------------------------------------------------------
# class FreqLinearAttention(nn.Module):
#     def __init__(self, dim, heads=8, drop=0., r=64):
#         super().__init__()
#         assert dim % heads == 0
#         self.h   = heads
#         self.d   = dim // heads  
#         self.r   = r            

#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.proj = nn.Linear(dim, dim)

#         scale = 1.0 / math.sqrt(self.d)
#         self.W = nn.Parameter(torch.randn(self.h, self.d, self.r // 2) * scale)
#         self.b = nn.Parameter(2*math.pi * torch.rand(self.h, self.r // 2))

#         self.drop = nn.Dropout(drop)
#         self.scale = (self.d ** -0.5)

#     def forward(self, x):
#         B, M, D = x.shape

#         # 1) rFFT
#         fx = torch.fft.rfft(x, norm='forward')
#         fx = torch.view_as_real(fx)  
#         fx = torch.cat((fx[..., 0], -fx[..., 1]), dim=-1)[..., :D]

#         # 2) Q,K,V
#         Q = self.q(fx)
#         K = self.k(fx)
#         V = self.v(fx)

#         def shp(t):
#             return t.view(B, M, self.h, self.d).transpose(1, 2) 
#         q, k, v = map(shp, (Q, K, V))

#         # 3) 随机特征映射 => 线性注意力
#         phi_q = random_fourier_features(q, self.W, self.b)
#         phi_k = random_fourier_features(k, self.W, self.b)

#         K_sum  = phi_k.sum(dim=2)  
#         KV_sum = torch.einsum('bhmr,bhmd->bhrd', phi_k, v)

#         denominator = torch.einsum('bhmr,bhr->bhm', phi_q, K_sum) 
#         numerator   = torch.einsum('bhmr,bhrd->bhmd', phi_q, KV_sum)

#         attn_out = numerator / (denominator.unsqueeze(-1) + 1e-6)

#         # 4) 多头合并
#         attn_out = attn_out.transpose(1, 2).reshape(B, M, D)
#         attn_out = self.proj(attn_out)

#         # 5) iFFT 回到时域
#         r, i = torch.chunk(attn_out, 2, dim=-1)
#         y = torch.fft.irfft(torch.complex(r, -i), n=D, norm='forward')
#         return y


# # -----------------------------------------------------------
# #   5. Transformer Block：FreqBlock（使用 FreqLinearAttention）
# # -----------------------------------------------------------
# class FreqBlock(nn.Module):
#     def __init__(self, dim, heads=8, mlp_ratio=4., drop=0., r=64):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn  = FreqLinearAttention(dim, heads, drop, r=r)
#         self.norm2 = nn.LayerNorm(dim)
#         hidden_dim = int(dim * mlp_ratio)
#         self.mlp   = nn.Sequential(
#             nn.Linear(dim, hidden_dim), 
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, dim),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x


# # -----------------------------------------------------------
# #   6. 整体网络  SimplePatchGNN (使用线性频域注意力) 
# #      【在进入 TTKMN 之前，先加 TE】
# # -----------------------------------------------------------
# class SimplePatchGNN(nn.Module):
#     """ 
#     T-TKMN + 频域(线性) Transformer + 线性聚合 + 解码
#     在进入 self.intra(...) 前，对 X_flat 注入一次时间编码
#     """
#     def __init__(self, args):
#         super().__init__()
#         self.device = args.device
#         self.N      = args.ndim
#         self.M      = args.npatch
#         self.L      = args.patch_size        
#         self.hid    = args.hid_dim
#         self.K      = 4                     
#         self.te_dim = args.te_dim

#         # ---- patch-内 ---
#         self.intra = TTKMN(self.K)   

#         # (新增) 用于把 TE (te_dim) 映射到 1 维, 便于与 X_flat 相加
#         self.te_proj_intra = nn.Linear(self.te_dim, 1)

#         # ---- patch-间（Transformer）---
#         self.pos = PositionalEncoding(self.hid)
#         self.blocks = nn.ModuleList([
#             FreqBlock(self.hid, heads=args.nhead, mlp_ratio=4., drop=0., r=64)
#             for _ in range(args.nlayer)
#         ])

#         # ---- 投影/聚合/解码 ---
#         self.patch_proj = nn.Linear(self.K + 1, self.hid)  # K维 + flag
#         self.agg        = nn.Linear(self.hid*self.M, self.hid)
#         self.dec        = nn.Sequential(
#             nn.Linear(self.hid + self.te_dim, self.hid), 
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hid, self.hid),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hid, 1)
#         )

#         # --- time-embedding ---
#         self.te_scale = nn.Linear(1, 1)
#         self.te_per   = nn.Linear(1, self.te_dim - 1)

#     def _TE(self, t): 
#         """
#         t: (..., 1)
#         返回: (..., te_dim)
#         """
#         return torch.cat([
#             self.te_scale(t), 
#             torch.sin(self.te_per(t))
#         ], dim=-1)

#     def forecasting(self, tp_pred, X, tp_true, mask=None):
#         """
#         X :      (B, M, L, N)
#         tp_true: (B, M, L, N)
#         mask:    (B, M, L, N)
#         tp_pred: (B, 1, Lp, 1) 或 (B, N, Lp, 1)
#         """
#         B,M,L,N = X.shape

#         # Print shapes of input data
       
       
     
        
#         if mask is None:
#             mask = torch.ones_like(X)

#         # 1) 变形 => (B*N*M, L, 1)
#         X_flat = X.permute(0,3,1,2).reshape(-1, L, 1)       
#         m_flat = mask.permute(0,3,1,2).reshape(-1, L, 1)
#         t_flat = tp_true.permute(0,3,1,2).reshape(-1, L, 1)

#         # 2) 对 t_flat 做时间编码 => (B*N*M, L, te_dim)
#         te_flat = self._TE(t_flat)

#         # 3) 投影到 1 维并与 X_flat 相加 => (B*N*M, L, 1)
#         te_1d   = self.te_proj_intra(te_flat)  
#         X_enh   = X_flat + te_1d  

#         # 4) 送入 TTKMN => (B*N*M, K+1)
#         Xf = self.intra(t_flat, X_enh, m_flat)

#         # 5) 恢复 => (B, N, M, K+1)
#         Xf = Xf.view(B, N, M, -1)

#         # 6) patch_proj => (B, N, M, hid)
#         feat = self.patch_proj(Xf)


#         # 7) 逐层 Transformer（layer > 0 时添加跨层残差）
#         feat = feat.reshape(B*N, M, self.hid)
#         feat = self.pos(feat)

#         for i, blk in enumerate(self.blocks):
#             if i == 0:
#                 feat = blk(feat)            # 第 0 层正常前向
#             else:
#                 feat = feat + blk(feat)     # layer > 0 : 输入 + 输出

#         feat = feat.reshape(B, N, M, self.hid)


#         # 8) 聚合 => (B, N, hid)
#         feat = self.agg(feat.reshape(B, N, -1))

#         # 9) 对预测时间点做解码
#         Lp = tp_pred.shape[-1]
#         h  = feat.unsqueeze(2).repeat(1, 1, Lp, 1)   # (B,N,Lp,hid)

#         te_p = self._TE(tp_pred.view(B,1,Lp,1).repeat(1,N,1,1)) # (B,N,Lp,te_dim)
#         out  = self.dec(torch.cat([h, te_p], dim=-1)) # => (B,N,Lp,1)
#         out  = out.squeeze(-1)                        # => (B,N,Lp)

#         # 10) 与原接口一致, 返回(1,B,Lp,N)
#         return out.unsqueeze(0).permute(0,1,3,2)

#     # forward = forecasting
#     def forward(self, tp_pred, X, tp_true, mask=None):
#         return self.forecasting(tp_pred, X, tp_true, mask)





import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------
# 1. TTKMN  ——  K-Gaussian kernel pooling over time
# -----------------------------------------------------------
class TTKMN(nn.Module):
    """K-Gaussian kernel pooling over time (per variable)."""
    def __init__(self, K: int = 4):
        super().__init__()
        self.K = K
        self.c          = nn.Parameter(torch.linspace(0, 1, K))   # centres
        self.log_alpha  = nn.Parameter(torch.zeros(K))             # log σ
        self.gate       = nn.Parameter(torch.zeros(K))             # kernel gates

    def forward(self, t, x, m):
        """
        t, x, m : (B*, L, 1)
        →        (B*, K + 1)
        """
        alpha = self.log_alpha.exp() + 1e-6                       # (K,)
        td    = t - self.c.view(1, 1, self.K)                     # (B*,L,K)
        w     = torch.exp(-0.5 * td**2 / alpha.view(1,1,self.K)**2) * m
        a     = w / (w.sum(1, keepdim=True) + 1e-8)               # attention
        h     = torch.einsum('blk,bld->bk', a, x)                 # (B*,K)
        h     = h * torch.sigmoid(self.gate)                      # gated
        flag  = (m.sum(1) > 0).float()                            # (B*,1)
        return torch.cat([h, flag], -1)                           # (B*,K+1)


# -----------------------------------------------------------
# 2. Positional Encoding  (变量维 N ≤ 512)
# -----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe  = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))               # (1,L,d)

    def forward(self, x):                                         # (B,L,d)
        return x + self.pe[:, :x.size(1)]


# -----------------------------------------------------------
# 3. 线性注意力（频域 + 随机特征映射）
# -----------------------------------------------------------
def rff(x, W, b):
    proj = torch.einsum('bhmd,hdR->bhmR', x, W) + b.unsqueeze(1).unsqueeze(2)
    return (torch.cat([torch.cos(proj), torch.sin(proj)], -1) /
            math.sqrt(proj.size(-1)))

class FreqLinearAttention(nn.Module):
    def __init__(self, dim, heads=8, r=64):
        super().__init__()
        assert dim % heads == 0
        self.h, self.d, self.r = heads, dim // heads, r
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        scale = 1.0 / math.sqrt(self.d)
        self.W = nn.Parameter(torch.randn(self.h, self.d, r // 2) * scale)
        self.b = nn.Parameter(2 * math.pi * torch.rand(self.h, r // 2))

    def forward(self, x):                                         # (B,M,D)
        B, M, D = x.shape
        fx = torch.fft.rfft(x, norm='forward')                    # (B,M,D/2+1)
        fx = torch.view_as_real(fx)                               # (B,M,D/2+1,2)
        fx = torch.cat((fx[..., 0], -fx[..., 1]), -1)[..., :D]    # (B,M,D)

        def split(t):                                             # → (B,h,M,d)
            return t.view(B, M, self.h, self.d).transpose(1, 2)
        q, k, v = map(split, (self.q(fx), self.k(fx), self.v(fx)))

        phi_q, phi_k = rff(q, self.W, self.b), rff(k, self.W, self.b)
        K_sum  = phi_k.sum(2)                                     # (B,h,R)
        KV_sum = torch.einsum('bhmr,bhmd->bhrd', phi_k, v)        # (B,h,R,d)
        out = torch.einsum('bhmr,bhrd->bhmd', phi_q, KV_sum) / \
              (torch.einsum('bhmr,bhr->bhm', phi_q, K_sum)
               .unsqueeze(-1) + 1e-6)
        out = out.transpose(1, 2).reshape(B, M, D)                # (B,M,D)
        out = self.proj(out)

        r, i = torch.chunk(out, 2, -1)
        return torch.fft.irfft(torch.complex(r, -i), n=D,
                               norm='forward')


class FreqBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4., r=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = FreqLinearAttention(dim, heads, r)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(True),
                                 nn.Linear(hidden, dim))

    def forward(self, x):                                         # (B,N,D)
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# -----------------------------------------------------------
# 4. SimplePatchGNN  (原始版本，含 pre_conv + TTKMN)
# -----------------------------------------------------------
class SimplePatchGNN(nn.Module):
    """
    输入:
        X        (B,L,N)   — 多变量序列
        tp_true  (B,L)     — 真实时间戳
        mask     (B,L,N)
        tp_pred  (B,Lp)    — 目标预测时间戳
    输出:
        (1,B,Lp,N)
    """
    def __init__(self, args):
        super().__init__()
        self.N   = args.ndim
        self.K   = args.K
        self.hid = args.hid_dim
        self.te = args.te_dim
        self.preconvdim = args.preconvdim


        self.intra        = TTKMN(self.K)
        self.te_proj1d    = nn.Linear(self.te, 1)

        self.pos          = PositionalEncoding(self.hid, max_len=self.N)
        self.blocks       = nn.ModuleList([FreqBlock(self.hid,
                                                     heads=args.nhead,
                                                     mlp_ratio=4.,
                                                     r=64)
                                            for _ in range(args.nlayer)])

        self.feat_proj    = nn.Linear(self.K + 1, self.hid)
        self.var_agg      = nn.Linear(self.hid, self.hid)
        self.te_per_sin = nn.Linear(1, (self.te - 1) // 2)
        self.te_per_cos = nn.Linear(1, self.te - 1 - ((self.te - 1) // 2))
        self.pre_conv = nn.Sequential(
            nn.Conv1d(1, self.preconvdim, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(self.preconvdim, 1, kernel_size=1)
        )

        self.decoder      = nn.Sequential(
            nn.Linear(self.hid + self.te, self.hid), nn.ReLU(True),
            nn.Linear(self.hid, self.hid),           nn.ReLU(True),
            nn.Linear(self.hid, 1)
        )

        self.te_scale = nn.Linear(1, 1)
        self.te_per   = nn.Linear(1, self.te - 1)

    # ---------- time embedding ----------
    def _TE(self, t):
        return torch.cat([
            self.te_scale(t),
            torch.sin(self.te_per_sin(t)),
            torch.cos(self.te_per_cos(t))
        ], -1)

    # ---------- forward ----------
    def forecasting(self, tp_pred, X, tp_true, mask=None):
        B, L, N = X.shape
        mask = torch.ones_like(X) if mask is None else mask
        print(X.shape)
        # broadcast 时间戳
        tp_true = tp_true[..., None].repeat(1, 1, N) if tp_true.dim() == 2 else tp_true
        tp_pred = tp_pred[..., None]                 if tp_pred.dim() == 2 else tp_pred

        # ----- 预卷积 -----
        Xf = X.transpose(1, 2).reshape(-1, 1, L)         # (B*N,1,L)
        Xf = self.pre_conv(Xf)           # (B*N,L,1)
        Xf = Xf.transpose(1, 2)
        Tf = tp_true.permute(0, 2, 1).reshape(-1, L, 1)

        Tf_min = Tf.min(dim=1, keepdim=True)[0]
        Tf_max = Tf.max(dim=1, keepdim=True)[0]
        Tf_normalized = (Tf - Tf_min) / (Tf_max - Tf_min + 1e-8)


        Mf = mask.permute(0, 2, 1).reshape(-1, L, 1)

        te  = self._TE(Tf)                               # (B*N,L,te)
        Xf_enh = Xf + self.te_proj1d(te)                 # 加时间编码
        
        
        z   = self.intra(Tf_normalized, Xf_enh, Mf)      # (B*N,K+1)
        z   = self.feat_proj(z).view(B, N, self.hid)     # (B,N,hid)

        # 变量维 Transformer
        z   = self.pos(z)
        for blk in self.blocks:
            z = blk(z)
        z   = self.var_agg(z).transpose(1, 2)            # (B,hid,N)

        # ---- 预测阶段 ----
        Lp  = tp_pred.shape[1]
        h   = z.unsqueeze(2).repeat(1, 1, Lp, 1)         # (B,hid,Lp,N)
        h   = h.permute(0, 3, 2, 1)                      # (B,N,Lp,hid)

        te_p = self._TE(tp_pred).unsqueeze(1).repeat(1, N, 1, 1)   # (B,N,Lp,te)

        y    = torch.cat([h, te_p], -1)            # (B,N,Lp,hid+te)
        y    = self.decoder(y).squeeze(-1)         # (B,N,Lp)

        return y.unsqueeze(0).permute(0, 1, 3, 2)  # (1,B,Lp,N)

    def forward(self, tp_pred, X, tp_true, mask=None):
        return self.forecasting(tp_pred, X, tp_true, mask)