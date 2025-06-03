# from typing import Optional

# from pathlib import Path
# import torch
# from torch import nn
# from spikingjelly.activation_based import surrogate, neuron, functional

 
# from module.positional_encoding import PositionEmbedding
# from module.spike_encoding import SpikeEncoder
# from module.spike_attention import Block

# tau = 2.0  # beta = 1 - 1/tau
# backend = "torch"
# detach_reset = True

# class ConvEncoder(nn.Module):
#     def __init__(self, output_size: int, kernel_size: int = 3):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=output_size,
#                 kernel_size=(1, kernel_size),
#                 stride=1,
#                 padding=(0, kernel_size // 2),
#             ),
#             nn.BatchNorm2d(output_size),
#         )
#         self.lif = neuron.LIFNode(
#             tau=tau,
#             step_mode="m",
#             detach_reset=detach_reset,
#             surrogate_function=surrogate.ATan(),
#         )

#     def forward(self, inputs: torch.Tensor):
#         # inputs: B, L, C
#         inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, C, L
#         enc = self.encoder(inputs)  # B, T, C, L
#         B, T, C, L = enc.shape
#         enc = enc.permute(1, 0, 2, 3)  # T, B, C, L
#         spks = self.lif(enc)  # T, B, C, L
#         return spks
    
    
 
# class Model(nn.Module):
#     _snn_backend = "spikingjelly"
    
#     def __init__(self, configs, patch_len=16, stride=8):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         d_ff: Optional[int] = None
#         depths: int = 2
#         common_thr: float = 1.0
#         self.seq_len = configs.seq_len
#         num_steps: int = 4
#         heads: int = 8
#         qkv_bias: bool = False
#         qk_scale: float = 0.125
#         input_size: Optional[int] = None
#         weight_file: Optional[Path] = None
#         encoder_type: Optional[str] = "conv"
#         self.d_model = configs.d_model
#         self.head = 8
#         self.d_ff = configs.d_ff #or dim * 4
#         self.T = num_steps
#         self.depths = depths

#         self.encoder = ConvEncoder
#         self.linear1 = nn.Linear(self.enc_in, self.d_model)
#         self.linear2 = nn.Linear(self.d_model, self.enc_in)
#         self.init_lif = neuron.LIFNode(
#             tau=tau,
#             step_mode="m",
#             detach_reset=detach_reset,
#             surrogate_function=surrogate.ATan(),
#             v_threshold=common_thr,
#             backend=backend,
#         )

#         self.blocks = nn.ModuleList(
#             [
#                 Block(
#                     length=self.seq_len,
#                     tau=tau,
#                     common_thr=common_thr,
#                     dim=self.d_model,
#                     d_ff=self.d_ff,
#                     heads=heads,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                 )
#                 for _ in range(depths)
#             ]
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)

#     def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         functional.reset_net(self)

#         x = self.encoder(x)  # B L C -> T B C L

#         x = self.init_lif(x)
#         x = self.linear1(x.permute(0, 1, 3, 2))

#         for blk in self.blocks:
#             x = blk(x)  # T B L D
#         out = x.mean(0)
#         out = self.linear2(out)
#         return out  # B L D, B D

#     @property
#     def output_size(self):
#         return self.dim

#     @property
#     def hidden_size(self):
#         return self.dim

from typing import Optional
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional
from module.spike_attention import Block
# 假设你仍沿用原有的 ConvEncoder 和 Block，实现不变
# 这里仅作引用演示
# from module.positional_encoding import PositionEmbedding
# from module.spike_encoding import SpikeEncoder
# from module.spike_attention import Block

tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True

class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        """
        inputs 形状: [B, L, C]
        1) 先 permute -> [B, 1, C, L]
        2) 经 self.encoder -> [B, output_size, C, L]
        3) 再 permute -> [T, B, C, L], 其中 T = out_channels(即 output_size)
           但在原逻辑中 T 是作为时序步长来用，这一点要注意和不规则数据的含义是否对得上
        """
        # B, L, C -> B, 1, C, L
        x = inputs.permute(0, 2, 1).unsqueeze(1)
        x = self.encoder(x)  # => B, out_channels, C, L
        # 这里 out_channels 相当于新的“时序”维度 T
        # 在原模型中被视为时间步; 也可以把它当成通道维度进行后续处理
        x = x.permute(1, 0, 2, 3)  # => T, B, C, L
        spks = self.lif(x)
        return spks


# class Block(nn.Module):
#     """
#     这里仅示意：你原本使用的 Block (spike_attention) 结构
#     """
#     def __init__(self, length, tau, common_thr, dim, d_ff, heads, qkv_bias, qk_scale):
#         super().__init__()
#         # 省略细节，假设内部是若干注意力/卷积操作 + LIF节点
#         self.tau = tau
#         self.dim = dim
#         # ... 其他初始化

#     def forward(self, x: torch.Tensor):
#         """
#         x: [T, B, L, D]
#         做某种时序/注意力处理后再输出
#         """
#         # 这里只是演示，不做实际处理
#         return x


class IrregularModel(nn.Module):
    _snn_backend = "spikingjelly"
    
    def __init__(self, args, patch_len=16, stride=8):
        super(IrregularModel, self).__init__()
        # self.task_name = args.task_name
        # self.pred_len = args.pred_len
        self.enc_in = 12  # 输入变量数
        self.seq_len = args.seq_len
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.depths = 2
        self.num_steps = 4
        self.heads = 8
        self.qkv_bias = False
        self.qk_scale = 0.125
        self.common_thr = 1.0

        # 原有编码器: ConvEncoder(out_channels = d_model)
        self.encoder = ConvEncoder(output_size=self.d_model)

        # 两个线性层，分别做升维和降维
        self.linear_in = nn.Linear(self.enc_in, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.enc_in)

        # 初始 LIF
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=self.common_thr,
            backend=backend,
        )

        # 堆叠若干 Block
        self.blocks = nn.ModuleList(
            [
                Block(
                    length=self.seq_len,
                    tau=tau,
                    common_thr=self.common_thr,
                    dim=self.d_model,
                    d_ff=self.d_ff,
                    heads=self.heads,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                )
                for _ in range(self.depths)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        X: torch.Tensor,       # [B, L, N]  多变量不规则时序
        tp_true: torch.Tensor, # [B, L]     对应 X 的实际时间戳
        tp_pred: torch.Tensor, # [B, Lp]    需要预测的目标时间戳
        mask: torch.Tensor=None
    ):
        """
        目标: 输出形状 (1, B, Lp, N)

        这里示例代码中，并没有对不规则时间戳做专门处理。
        实际中, 你可能需要:
            1) 根据 (tp_true, X, mask) 先做插值或差值编码 (Delta t)
            2) 或者在神经网络内部对时间戳做 Positional Embedding / GRU-D 门控等
        """
        # 1) 重置 SNN 网络内部状态
        functional.reset_net(self)

        B, L, N = X.shape
        Lp = tp_pred.shape[1]  # 需要预测的时间点数

        # [可选] 如果要直接根据 mask 把缺失位置置零或其他处理
        # X_masked = X * mask  # 简单示例；实际上最好保留一个 embedding 告知模型哪些点是缺失

        # 2) 先用线性层升维: [B, L, N] -> [B, L, d_model]
        X_in = self.linear_in(X)

        # 3) 调用 conv encoder 前，需要把 X_in 视作 (B, L, C):
        # 这与上面 ConvEncoder.forward(inputs: [B, L, C]) 的签名匹配
        spks = self.encoder(X_in)  # => shape [T, B, C, L]  (T=d_model, C=?)

        # 4) 初始 LIF 处理: 这里 spks 依旧是 [T, B, C, L]
        spks = self.init_lif(spks)

        # 5) 注意：ConvEncoder 的最后 permute 使得 T = out_channels = self.d_model
        #    spks: [T, B, C, L] => 需要变为 [T, B, L, D]
        #    取决于你在 Block 里期望的输入形状. 
        #    这里假设 C 就是特征维度, L 是时间步 => 你可能需要再 permute 一下:
        spks = spks.permute(0, 1, 3, 2)  # => [T, B, L, C], 令 C = d_model

        # 6) 堆叠 Block
        for blk in self.blocks:
            spks = blk(spks)  # 期望输入/输出维度都是 [T, B, L, D]

        # 7) 简单汇聚: 这里原本 x.mean(0) => [B, L, D]
        out = spks.mean(dim=0)  # => shape [B, L, D]

        # 8) 将特征维度投影回原变量数 N, => [B, L, N]
        out = self.linear_out(out)

        # ------------
        # ★ 关键: 如何将 [B, L, N] 映射到 [B, Lp, N] (对应 tp_pred)?
        #   - 若是常规 seq2seq, 通常 L -> pred_len
        #   - 在不规则场景下, 需要插值或对未来时间戳做解码
        #   - 这里只给示例: 简单地取后 Lp 步(或其它方式), 或对 (tp_pred - tp_true) 做额外插值
        # ------------

        # (示例) 如果 L >= Lp, 可以截取最后 Lp 步:
        # out_pred: [B, Lp, N]
        if L >= Lp:
            out_pred = out[:, -Lp:, :]
        else:
            # 如果 L < Lp，需要补零或插值、或者用额外Decoder来预测更多步
            # 这里仅演示: 先简单 repeat 一下
            repeat_num = Lp - L
            # 拼接
            out_pred = torch.cat([out, out[:, -1:, :].repeat(1, repeat_num, 1)], dim=1)

        # 9) 最终加一个 “1” 维度 => [1, B, Lp, N]
        out_pred = out_pred.unsqueeze(0)

        return out_pred

    @property
    def output_size(self):
        return self.d_model

    @property
    def hidden_size(self):
        return self.d_model
