"""Frozen NoGate reference extracted from the user's course-project SourceCode.zip.

Do not refactor: used only to verify LTRR forward/gradient equivalence.
Original path: src/models/SepReformer_SGLR_NoGate_Libri2Mix_16K/modules/module.py
Original module SHA-256: caf5b25be5b20a26f1fbabce89af4c52f6742cad4a7d82dadb93af75ef4a1fdd
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

class SpeakerGuidedMossGCU(torch.nn.Module):
    """
    GCU-style refinement with optional speaker-guided gate.

    Full SGLR:
        U = ConvU(Z)
        V = ConvU(Z)
        M = DenseDilatedFSMN(V)
        gate = sigmoid(PWConv([Z, speaker_contrast]))
        O = Z + gate * (U * M)

    NoGate ablation:
        O = Z + (U * M)

    Input/Output: [B*num_spks, C, T]
    """
    def __init__(
        self,
        channels: int,
        num_spks: int,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout_rate: float = 0.05,
        use_speaker_gate: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_spks = num_spks
        self.use_speaker_gate = bool(use_speaker_gate)

        self.conv_u = ConvU(channels, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.conv_v = ConvU(channels, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.memory = DenseDilatedFSMN1D(
            channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout_rate=dropout_rate,
        )

        if self.use_speaker_gate:
            self.speaker_gate = torch.nn.Sequential(
                torch.nn.Conv1d(channels * 2, channels, kernel_size=1),
                torch.nn.Sigmoid(),
            )
        else:
            self.speaker_gate = None

        self.dropout = torch.nn.Dropout(dropout_rate)

    def _speaker_contrast(self, z: torch.Tensor):
        BJ, C, T = z.shape
        if BJ % self.num_spks != 0:
            raise RuntimeError(
                f"SpeakerGuidedMossGCU expected BJ divisible by num_spks, got BJ={BJ}, num_spks={self.num_spks}"
            )

        B = BJ // self.num_spks
        z_spk = z.view(B, self.num_spks, C, T)

        if self.num_spks == 1:
            contrast = torch.zeros_like(z_spk)
        else:
            other_mean = (z_spk.sum(dim=1, keepdim=True) - z_spk) / (self.num_spks - 1)
            contrast = z_spk - other_mean

        return contrast.view(BJ, C, T)

    def forward(self, z: torch.Tensor):
        u = self.conv_u(z)
        v = self.conv_v(z)
        m = self.memory(v)

        if self.use_speaker_gate:
            contrast = self._speaker_contrast(z)
            gate = self.speaker_gate(torch.cat([z, contrast], dim=1))
            refinement = gate * (u * m)
        else:
            refinement = u * m

        refinement = self.dropout(refinement)
        return z + refinement

class SGLR(torch.nn.Module):
    """
    SepReformer-SGLR: Speaker-Guided Lightweight Recurrent Refinement.

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
        use_speaker_gate: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_spks = num_spks
        self.bottleneck_channels = bottleneck_channels
        self.use_speaker_gate = bool(use_speaker_gate)

        self.bottleneck = torch.nn.Sequential(
            ChannelLayerNorm(in_channels),
            torch.nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
            torch.nn.PReLU(),
        )

        self.gcu = SpeakerGuidedMossGCU(
            channels=bottleneck_channels,
            num_spks=num_spks,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout_rate=dropout_rate,
            use_speaker_gate=use_speaker_gate,
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
