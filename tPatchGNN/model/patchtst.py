import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTimeEmbedding(nn.Module):
    """
    针对不规则时间步的可学习时间嵌入：
    输入: (B, L) 或 (B, L, 1)，每个位置是真实时间(可以是 float)。
    输出: (B, L, time_embed_dim)
    """
    def __init__(self, time_embed_dim=8):
        super(LearnableTimeEmbedding, self).__init__()
        self.time_embed_dim = time_embed_dim
        # 这里给个简单的两层 MLP，你也可以自行修改/加SinCos等
        self.mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, t):
        # t: (..., 1) 或 (...), 这里保证最后一维是时间
        # 若 t 形状为 (B, L)，先扩到 (B, L, 1) 再过 mlp
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # => (B, L, 1)
        return self.mlp(t)  # => (B, L, time_embed_dim)


class SimpleTransformerEncoder(nn.Module):
    """
    用于对单个 patch 内的 (L) 个时间步进行编码的 TransformerEncoder。
    如果需要多层堆叠，可以外层自己用 nn.ModuleList 重复调用，或直接把num_layers>1。
    """
    def __init__(self, d_model=64, nhead=4, dim_feedforward=256, num_layers=1):
        super(SimpleTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x shape: (B, L, d_model)
        返回: (B, L, d_model)
        """
        return self.transformer(x)


class IrregularTimeTransformer(nn.Module):
    """
    针对不规则时间序列的简化 Transformer 预测模型：
    1) 对每个 patch (B, L, N)，以及对应不规则时间 truth_time_steps (B, L)，进行编码；
    2) 得到每个 patch 的表示后，聚合 M 个 patch；
    3) 对预测时间步 time_steps_to_predict 同样做时间嵌入并解码。
    """
    def __init__(self, args):
        super(IrregularTimeTransformer, self).__init__()
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.M = args.npatch
        self.device = args.device

        # 时间嵌入维度
        self.time_embed_dim = args.te_dim

        # 将 (N + time_embed_dim) -> hid_dim 的线性映射
        self.input_proj = nn.Linear(self.N + self.time_embed_dim, self.hid_dim)

        # 这里定义一个针对单个 patch 内时间维度(L)的 TransformerEncoder
        self.patch_encoder = SimpleTransformerEncoder(
            d_model=self.hid_dim,
            nhead=args.nhead,
            dim_feedforward=args.hid_dim,   # 中间层大小，可自定义
            num_layers=args.tf_layer      # 堆叠层数
        )

        # 可学习的时间嵌入，用来处理 (B, L) 形状的真实时间与预测时间
        self.time_embed = LearnableTimeEmbedding(time_embed_dim=self.time_embed_dim)

        # 对每个 patch 编码后的向量进行聚合 (比如 mean-pooling) 得到 (B, hid_dim)
        # 然后再聚合 M 个 patch (这里同样用 mean-pooling 或 Linear 等)
        self.patch_agg = nn.Linear(self.hid_dim, self.hid_dim)  # 示例：最后可再加一层线性
        # 也可以改成CNN之类，这里仅演示最简单方式

        # M个patch聚合：把 M 个 (B, hid_dim) 先堆在一起 (B, M, hid_dim)，再用一个简单平均或线性
        self.cross_patch_linear = nn.Linear(self.hid_dim, self.hid_dim)

        # Decoder：给定最终的 (B, hid_dim) 与要预测的时间步 (B, Lp)，
        #          做时间嵌入后合并，然后输出 (B, N, Lp, 1)
        self.decoder_hidden = nn.Linear(self.hid_dim + self.time_embed_dim, self.hid_dim)
        self.decoder_out = nn.Linear(self.hid_dim, 1)  # 最终输出 1 维

    def encode_patch(self, X_patch, T_patch):
        """
        编码单个 patch：
        X_patch: (B, L, N)   - 该 patch 下的输入序列
        T_patch: (B, L, 1)   - 该 patch 对应的不规则时间
        返回: (B, hid_dim)，该 patch 在时间维度上的聚合表示
        """
        B, L, N = X_patch.shape
        # 1) 时间嵌入
        #    T_patch => (B, L, time_embed_dim)
        time_emb = self.time_embed(T_patch)  # (B, L, time_embed_dim)

        # 2) 拼接输入特征: cat([X_patch, time_emb], -1) => (B, L, N + time_embed_dim)
        x_with_time = torch.cat([X_patch, time_emb], dim=-1)

        # 3) 输入线性投影 -> (B, L, hid_dim)
        x_proj = self.input_proj(x_with_time)

        # 4) TransformerEncoder 编码 => (B, L, hid_dim)
        x_enc = self.patch_encoder(x_proj)

        # 5) 在时间维度 L 上做一个 pooling 或聚合 => (B, hid_dim)
        x_patch_rep = x_enc.mean(dim=1)
        # 也可以使用 x_enc[:, -1, :] 或者其他聚合方式

        # 6) 再用一个线性或非线性变化 => (B, hid_dim)
        x_patch_rep = self.patch_agg(x_patch_rep)
        return x_patch_rep


    
    

    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        """
        与原模型接口相同:
        time_steps_to_predict : (B, Lp)
        X : (B, M, L, N)
        truth_time_steps : (B, M, L, N)
        mask : (B, M, L, N) (可选)
        
        返回: (1, B, Lp, N)
        """
        B, M, L, N = X.shape
        # 把每个 patch 分开处理
        patch_reps = []
        for m in range(M):
            # 当前 patch 的数据与时间: (B, L, N), (B, L, N)
            X_patch = X[:, m, :, :]  # (B, L, N)
            T_patch = truth_time_steps[:, m, :, :]  # (B, L, N)
            # 若时间对所有 N 通道相同，可以只取 T_patch[..., :1] => (B, L, 1)
            # 否则如果每个通道不一样，也可尝试把 N 个时间分别嵌入后再与 X 拼接
            T_patch_1d = T_patch[..., :1]  # (B, L, 1)，示例只用第0通道的时间

            # 调用单个 patch 的编码
            rep = self.encode_patch(X_patch, T_patch_1d)  # (B, hid_dim)
            patch_reps.append(rep)

        # 将 M 个 patch 的表示拼合: (B, M, hid_dim)
        patch_reps = torch.stack(patch_reps, dim=1)

        # 再做一次跨 patch 的聚合 (示例：平均 + Linear)
        # 你也可以使用小型 Transformer 来在 M 上做序列建模
        patch_mean = patch_reps.mean(dim=1)  # (B, hid_dim)
        final_rep = self.cross_patch_linear(patch_mean)  # (B, hid_dim)

        # =============== Decoder 阶段 ===============
        # 要预测的时间步 time_steps_to_predict: (B, Lp)
        # 做时间嵌入 => (B, Lp, time_embed_dim)
        T_pred_emb = self.time_embed(time_steps_to_predict)  # (B, Lp, time_embed_dim)

        # 将 final_rep (B, hid_dim) broadcast 到 (B, Lp, hid_dim)
        # 然后与 T_pred_emb 拼一起 => (B, Lp, hid_dim + time_embed_dim)
        Lp = time_steps_to_predict.size(-1)
        final_rep_expanded = final_rep.unsqueeze(1).expand(-1, Lp, -1)  # (B, Lp, hid_dim)
        decoder_inp = torch.cat([final_rep_expanded, T_pred_emb], dim=-1)  # (B, Lp, hid_dim+time_embed_dim)

        # 过一层隐层 => (B, Lp, hid_dim)
        dec_h = self.decoder_hidden(decoder_inp)
        dec_h = F.relu(dec_h)

        # 最终输出 => (B, Lp, 1)
        dec_out = self.decoder_out(dec_h)  # (B, Lp, 1)

        # 还需要变形到 (1, B, Lp, N) 以保持与原模型一致
        # 这里要注意 N: 若需要对每条通道输出不同预测，可在这里再做 expand。
        # 比如如果你要对每条通道都预测，可以 expand 到 (B, N, Lp) 再 reshape。
        # 简单起见，这里直接对所有通道统一输出 => (B, Lp, 1)，再 broadcast 到 (B, Lp, N)
        # 当然也可为每个通道单独建模。
        dec_out = dec_out.expand(-1, -1, N)  # => (B, Lp, N)

        # 最终 => (1, B, Lp, N)
        outputs = dec_out.permute(0, 1, 2).unsqueeze(0)  # => (1, B, Lp, N)
        return outputs
    
    def forward(self, time_steps_to_predict, X, truth_time_steps, mask = None):
        return self.forecasting(time_steps_to_predict, X, truth_time_steps, mask)