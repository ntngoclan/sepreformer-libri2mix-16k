"""Original course-project NoGate architecture, renamed LTRR.

Source: SourceCode.zip, SepReformer_SGLR_NoGate_Libri2Mix_16K/modules/module.py.
Only the unused speaker-gate path is removed. Tensor names inside the block,
normalization epsilon, activation functions, dropout and residuals are preserved.
"""
import torch

class ChannelLayerNorm(torch.nn.Module):
    """
    LayerNorm cho tensor dạng [B, C, T].
    PyTorch LayerNorm chuẩn hóa chiều cuối, nên cần transpose sang [B, T, C].
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor):
        # [B, C, T] -> [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        return x

class ConvU(torch.nn.Module):
    """
    Conv-U branch lấy cảm hứng từ MossFormer2.
    Input/Output: [B*num_spks, C, T]
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout_rate: float = 0.05):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.norm = ChannelLayerNorm(channels)
        self.linear = torch.nn.Conv1d(channels, channels, kernel_size=1)
        self.act = torch.nn.SiLU()
        self.dconv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        residual = x
        y = self.norm(x)
        y = self.linear(y)
        y = self.act(y)
        y = self.dconv(y)
        y = self.dropout(y)
        return residual + y

class DenseDilatedFSMN1D(torch.nn.Module):
    """
    Lightweight Dense Dilated FSMN-like memory.
    Giữ tinh thần của MossFormer2: dilation + dense connection + memory accumulation,
    nhưng dùng Conv1d depthwise để nhẹ và phù hợp với SepReformer feature [B*num_spks, C, T].
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.dilations = list(dilations)
        self.ffn_in = torch.nn.Sequential(
            ChannelLayerNorm(channels),
            torch.nn.Conv1d(channels, channels, kernel_size=1),
            torch.nn.PReLU(),
        )

        self.proj_layers = torch.nn.ModuleList()
        self.memory_layers = torch.nn.ModuleList()

        for idx, dilation in enumerate(self.dilations):
            dense_in_channels = channels * (idx + 1)
            self.proj_layers.append(torch.nn.Conv1d(dense_in_channels, channels, kernel_size=1))

            padding = dilation * (kernel_size - 1) // 2
            self.memory_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=channels,
                        bias=True,
                    ),
                    torch.nn.PReLU(),
                    torch.nn.Dropout(dropout_rate),
                )
            )

        self.out_proj = torch.nn.Conv1d(channels * len(self.dilations), channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0 = self.ffn_in(x)
        states = []

        for idx in range(len(self.dilations)):
            dense_input = torch.cat([x0] + states, dim=1)
            y = self.proj_layers[idx](dense_input)
            y = self.memory_layers[idx](y)
            states.append(y)

        memory = torch.cat(states, dim=1)
        memory = self.out_proj(memory)
        return memory


class TemporalInteraction(torch.nn.Module):
    """Two Conv-U branches with multiplicative dilated memory, no speaker gate."""
    def __init__(self, channels, num_spks, kernel_size=3,
                 dilations=(1, 2, 4, 8), dropout_rate=0.05):
        super().__init__()
        self.channels = channels
        self.num_spks = num_spks
        self.conv_u = ConvU(channels, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.conv_v = ConvU(channels, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.memory = DenseDilatedFSMN1D(
            channels, kernel_size=kernel_size, dilations=dilations,
            dropout_rate=dropout_rate,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, z):
        u = self.conv_u(z)
        v = self.conv_v(z)
        m = self.memory(v)
        refinement = self.dropout(u * m)
        return z + refinement

class LTRR(torch.nn.Module):
    """
    Lightweight Temporal Residual Refinement: the original NoGate block.

    Đặt sau Separator/Reconstruction Decoder và trước OutputLayer.
    Input : [B*num_spks, F, T]
    Output: [B*num_spks, F, T]
    """
    def __init__(
        self,
        in_channels: int,
        num_spks: int,
        bottleneck_channels: int = 64,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout_rate: float = 0.05,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_spks = num_spks
        self.bottleneck_channels = bottleneck_channels

        self.bottleneck = torch.nn.Sequential(
            ChannelLayerNorm(in_channels),
            torch.nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
            torch.nn.PReLU(),
        )

        self.gcu = TemporalInteraction(
            channels=bottleneck_channels,
            num_spks=num_spks,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout_rate=dropout_rate,
        )

        self.output_layer = torch.nn.Sequential(
            ChannelLayerNorm(bottleneck_channels),
            torch.nn.Conv1d(bottleneck_channels, in_channels, kernel_size=1),
        )

        # Khởi tạo nhỏ để không phá output của SepReformer ở đầu training.
        self.alpha = torch.nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, y: torch.Tensor):
        z = self.bottleneck(y)
        z = self.gcu(z)
        refinement = self.output_layer(z)
        return y + self.alpha * refinement
