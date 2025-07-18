import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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



class KFNet(nn.Module):
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
         
        
        
        tp_true = tp_true[..., None].repeat(1, 1, N) if tp_true.dim() == 2 else tp_true
        tp_pred = tp_pred[..., None]                 if tp_pred.dim() == 2 else tp_pred

        
        
        Xf = X.transpose(1, 2).reshape(-1, 1, L)         # (B*N,1,L)
        Xf = self.pre_conv(Xf)           # (B*N,L,1)
        Xf = Xf.transpose(1, 2)
        Tf = tp_true.permute(0, 2, 1).reshape(-1, L, 1)

        Tf_min = Tf.min(dim=1, keepdim=True)[0]
        Tf_max = Tf.max(dim=1, keepdim=True)[0]
        Tf_normalized = (Tf - Tf_min) / (Tf_max - Tf_min + 1e-8)


        Mf = mask.permute(0, 2, 1).reshape(-1, L, 1)

        te  = self._TE(Tf)                               # (B*N,L,te)
        Xf_enh = Xf + self.te_proj1d(te)                 
        
        
        z   = self.intra(Tf_normalized, Xf_enh, Mf)      # (B*N,K+1)
        z   = self.feat_proj(z).view(B, N, self.hid)     # (B,N,hid)

        
        
        z   = self.pos(z)
        for blk in self.blocks:
            z = blk(z)
        z   = self.var_agg(z).transpose(1, 2)            # (B,hid,N)

        
        
        Lp  = tp_pred.shape[1]
        h   = z.unsqueeze(2).repeat(1, 1, Lp, 1)         # (B,hid,Lp,N)
        h   = h.permute(0, 3, 2, 1)                      # (B,N,Lp,hid)

        te_p = self._TE(tp_pred).unsqueeze(1).repeat(1, N, 1, 1)   # (B,N,Lp,te)

        y    = torch.cat([h, te_p], -1)            # (B,N,Lp,hid+te)
        y    = self.decoder(y).squeeze(-1)         # (B,N,Lp)

        return y.unsqueeze(0).permute(0, 1, 3, 2)  # (1,B,Lp,N)

    def forward(self, tp_pred, X, tp_true, mask=None):
        return self.forecasting(tp_pred, X, tp_true, mask)