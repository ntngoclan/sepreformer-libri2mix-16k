import torch

from utils.decorators import *
from .network import *


class AudioEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int, bias: bool):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1) # [T] - >[1, T] OR [B, T] -> [B, 1, T]
        x = self.conv1d(x)
        x = self.gelu(x)
        return x
    
class FeatureProjector(torch.nn.Module):
    def __init__(self, num_channels: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1e-8)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.conv1d(x)
        return x


class ChannelLayerNorm(torch.nn.Module):
    """Layer normalization over channels for a [B, C, T] tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        return x.transpose(1, 2).contiguous()


class PARRConvU(torch.nn.Module):
    """Local Conv-U branch used by the shared PARR temporal core."""

    def __init__(self, channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("PARR kernel_size must be odd to preserve time length.")

        self.norm = ChannelLayerNorm(channels)
        self.pointwise = torch.nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = torch.nn.SiLU()
        self.depthwise = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        branch = self.norm(x)
        branch = self.pointwise(branch)
        branch = self.activation(branch)
        branch = self.depthwise(branch)
        branch = self.dropout(branch)
        return x + branch


class DenseDilatedFSMN1D(torch.nn.Module):
    """Dense dilated temporal memory used by PARR."""

    def __init__(self, channels: int, kernel_size: int, dilations, dropout_rate: float):
        super().__init__()
        if not dilations:
            raise ValueError("PARR dilations must contain at least one value.")

        self.dilations = tuple(dilations)
        self.input_projection = torch.nn.Sequential(
            ChannelLayerNorm(channels),
            torch.nn.Conv1d(channels, channels, kernel_size=1),
            torch.nn.PReLU(),
        )
        self.dense_projections = torch.nn.ModuleList()
        self.memory_layers = torch.nn.ModuleList()

        for index, dilation in enumerate(self.dilations):
            self.dense_projections.append(
                torch.nn.Conv1d(channels * (index + 1), channels, kernel_size=1)
            )
            self.memory_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=dilation * (kernel_size - 1) // 2,
                        dilation=dilation,
                        groups=channels,
                    ),
                    torch.nn.PReLU(),
                    torch.nn.Dropout(dropout_rate),
                )
            )

        self.output_projection = torch.nn.Conv1d(
            channels * len(self.dilations), channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor):
        initial = self.input_projection(x)
        states = []
        for dense_projection, memory_layer in zip(self.dense_projections, self.memory_layers):
            dense_input = torch.cat([initial] + states, dim=1)
            states.append(memory_layer(dense_projection(dense_input)))
        return self.output_projection(torch.cat(states, dim=1))


class PARRTemporalCore(torch.nn.Module):
    """Shared PARR temporal core with channel-temporal correction control."""

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        dilations,
        dropout_rate: float,
    ):
        super().__init__()
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
            torch.nn.PReLU(),
            # Keep the controller inputs on comparable scales: the correction
            # path is normalized below, so the bottleneck feature must also be
            # normalized before both tensors are concatenated.
            ChannelLayerNorm(bottleneck_channels),
        )
        self.conv_u = PARRConvU(bottleneck_channels, kernel_size, dropout_rate)
        self.conv_v = PARRConvU(bottleneck_channels, kernel_size, dropout_rate)
        self.memory = DenseDilatedFSMN1D(
            bottleneck_channels, kernel_size, dilations, dropout_rate
        )
        self.correction_norm = ChannelLayerNorm(bottleneck_channels)

        # A_r = sigmoid(G([Z_r, R_r])) with one value per bottleneck channel
        # and time step. This lets PARR select both when and which latent
        # features need refinement while preserving the original PARR flow.
        self.temporal_controller = torch.nn.Conv1d(
            2 * bottleneck_channels, bottleneck_channels, kernel_size=1
        )
        # A neutral controller starts with A_r(c, t) = sigmoid(0) = 0.5.
        # Together with gamma_r = 0 this retains an exact identity mapping and
        # avoids an arbitrary random channel/time preference at initialization.
        torch.nn.init.zeros_(self.temporal_controller.weight)
        torch.nn.init.zeros_(self.temporal_controller.bias)
        self.output_projection = torch.nn.Conv1d(
            bottleneck_channels, in_channels, kernel_size=1
        )

    def forward(self, normalized_stage_output: torch.Tensor):
        z = self.bottleneck(normalized_stage_output)
        u = self.conv_u(z)
        v = self.conv_v(z)
        memory = self.memory(v)

        # Pure correction path: R_r = Norm(U_r * M_r), without Z_r + R_r.
        residual_correction = self.correction_norm(u * memory)
        controller = torch.sigmoid(
            self.temporal_controller(torch.cat([z, residual_correction], dim=1))
        )
        # controller and residual_correction are both [BJ, C_b, T_r]; the
        # modulation is therefore element-wise, with no channel broadcasting.
        return self.output_projection(controller * residual_correction)


class ProgressiveAdaptiveResidualRefinement(torch.nn.Module):
    """PARR with a shared core and stage-specific normalization/scaling."""

    def __init__(
        self,
        num_stages: int,
        in_channels: int,
        bottleneck_channels: int = 64,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8),
        dropout_rate: float = 0.05,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.stage_norms = torch.nn.ModuleList(
            [ChannelLayerNorm(in_channels) for _ in range(num_stages)]
        )
        self.stage_scales = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(float(gamma_init))) for _ in range(num_stages)]
        )
        self.shared_temporal_core = PARRTemporalCore(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout_rate=dropout_rate,
        )

    def forward(self, stage_output: torch.Tensor, stage_index: int):
        if not 0 <= stage_index < self.num_stages:
            raise IndexError(
                f"PARR stage_index must be in [0, {self.num_stages}), got {stage_index}."
            )
        normalized = self.stage_norms[stage_index](stage_output)
        delta = self.shared_temporal_core(normalized)
        return stage_output + self.stage_scales[stage_index] * delta


class Separator(torch.nn.Module):
    def __init__(self, num_stages: int, relative_positional_encoding: dict, enc_stage: dict, spk_split_stage: dict, simple_fusion:dict, dec_stage: dict, parr: dict):
        super().__init__()
        
        class RelativePositionalEncoding(torch.nn.Module):
            def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
                super().__init__()
                self.in_channels = in_channels
                self.num_heads = num_heads
                self.embedding_dim = self.in_channels // self.num_heads
                self.maxlen = maxlen
                self.pe_k = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim)
                self.pe_v = torch.nn.Embedding(num_embeddings=2*maxlen, embedding_dim=self.embedding_dim) if embed_v else None
            
            def forward(self, pos_seq: torch.Tensor):
                pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
                pos_seq += self.maxlen
                pe_k_output = self.pe_k(pos_seq)
                pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
                return pe_k_output, pe_v_output
        
        class SepEncStage(torch.nn.Module):
            def __init__(self, global_blocks: dict, local_blocks: dict, down_conv_layer: dict, down_conv=True):
                super().__init__()
                                
                class DownConvLayer(torch.nn.Module):
                    def __init__(self, in_channels: int, samp_kernel_size: int):
                        """Construct an EncoderLayer object."""
                        super().__init__()
                        self.down_conv = torch.nn.Conv1d(
                            in_channels=in_channels, out_channels=in_channels, kernel_size=samp_kernel_size, stride=2, padding=(samp_kernel_size-1)//2, groups=in_channels)
                        self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                        self.gelu = torch.nn.GELU()
                    
                    def forward(self, x: torch.Tensor):
                        x = x.permute([0, 2, 1])
                        x = self.down_conv(x)
                        x = self.BN(x)
                        x = self.gelu(x)
                        x = x.permute([0, 2, 1])
                        return x
                
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                
                self.downconv = DownConvLayer(**down_conv_layer) if down_conv == True else None
                
            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                
                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                
                skip = x
                if self.downconv:
                    x = x.permute(0, 2, 1).contiguous()
                    x = self.downconv(x)
                    x = x.permute(0, 2, 1).contiguous()
                # [BK, S, N]
                return x, skip
        
        class SpkSplitStage(torch.nn.Module):
            def __init__(self, in_channels: int, num_spks: int):
                super().__init__()
                self.linear = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, 4*in_channels*num_spks, kernel_size=1),
                    torch.nn.GLU(dim=-2),
                    torch.nn.Conv1d(2*in_channels*num_spks, in_channels*num_spks, kernel_size=1))
                self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)
                self.num_spks = num_spks
                
            def forward(self, x: torch.Tensor):
                x = self.linear(x)
                B, _, T = x.shape
                x = x.view(B*self.num_spks,-1, T).contiguous()
                x = self.norm(x)
                return x
        
        class SepDecStage(torch.nn.Module):
            def __init__(self, num_spks: int, global_blocks: dict, local_blocks: dict, spk_attention: dict):
                super().__init__()
                
                self.g_block_1 = GlobalBlock(**global_blocks)
                self.l_block_1 = LocalBlock(**local_blocks)
                self.spk_attn_1 = SpkAttention(**spk_attention)
                
                self.g_block_2 = GlobalBlock(**global_blocks)
                self.l_block_2 = LocalBlock(**local_blocks)
                self.spk_attn_2 = SpkAttention(**spk_attention)
                
                self.g_block_3 = GlobalBlock(**global_blocks)
                self.l_block_3 = LocalBlock(**local_blocks)
                self.spk_attn_3 = SpkAttention(**spk_attention)
                
                self.num_spk = num_spks
            
            def forward(self, x: torch.Tensor, pos_k: torch.Tensor):
                '''
                x: [B, N, T]
                '''
                # [BS, K, H]
                x = self.g_block_1(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_1(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_1(x, self.num_spk)
                
                x = self.g_block_2(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_2(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_2(x, self.num_spk)
                
                x = self.g_block_3(x, pos_k)
                x = x.permute(0, 2, 1).contiguous()
                x = self.l_block_3(x)
                x = x.permute(0, 2, 1).contiguous()
                x = self.spk_attn_3(x, self.num_spk)
                
                skip = x
                
                return x, skip
        
        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(**relative_positional_encoding)
        
        # Temporal Contracting Part
        self.enc_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.enc_stages.append(SepEncStage(**enc_stage, down_conv=True))
        
        self.bottleneck_G = SepEncStage(**enc_stage, down_conv=False)
        self.spk_split_block = SpkSplitStage(**spk_split_stage)
        
        # Temporal Expanding Part
        self.simple_fusion = torch.nn.ModuleList([])
        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.simple_fusion.append(torch.nn.Conv1d(in_channels=simple_fusion['out_channels']*2,out_channels=simple_fusion['out_channels'], kernel_size=1))
            self.dec_stages.append(SepDecStage(**dec_stage))
        self.parr = ProgressiveAdaptiveResidualRefinement(
            num_stages=self.num_stages,
            **parr,
        )
    
    def forward(self, input: torch.Tensor):
        '''input: [B, N, L]'''
        # feature projection
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        # Temporal Contracting Part
        pos_seq = torch.arange(0, len_x//2**self.num_stages).long().to(x.device)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)
        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip_ = self.spk_split_block(skip_)
            skip.append(skip_)
        x, _ = self.bottleneck_G(x, pos_k)
        x = self.spk_split_block(x) # B, 2F, T
        
        each_stage_outputs = []
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = torch.nn.functional.interpolate(
                x, size=skip[idx_en].shape[-1], mode="nearest"
            )
            x = torch.cat([x,skip[idx_en]],dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x, pos_k)
            # PARR is applied after Global/Local/Cross-Speaker processing.
            # The refined tensor feeds the next stage (and its existing aux head).
            x = self.parr(x, stage_index=idx)
        
        last_stage_output = x 
        return last_stage_output, each_stage_outputs
    
    def pad_signal(self, input: torch.Tensor):
        #  (B, T) or (B, 1, T)
        if input.dim() == 1: input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]: raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        batch_size = input.size(0)  
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe//L + 1)*L
        rest = 0 if nframe%L == 0 else padded_len - nframe
        if rest > 0:
            pad = input.new_zeros(batch_size, ndim, rest)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_spks: int, masking: bool = False):
        super().__init__()
        # feature expansion back
        self.masking = masking
        self.spe_block = Masking(in_channels, Activation_mask="ReLU", concat_opt=None)
        self.num_spks = num_spks
        self.end_conv1x1 = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 4*out_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2*out_channels, in_channels))
            
    def forward(self, x: torch.Tensor, input: torch.Tensor):
        x = x[...,:input.shape[-1]]
        x = x.permute([0, 2, 1])
        x = self.end_conv1x1(x)
        x = x.permute([0, 2, 1])
        B, N, L = x.shape
        B = B // self.num_spks
        
        if self.masking:
            input = input.expand(self.num_spks, B, N, L).transpose(0,1).contiguous()
            input = input.view(B*self.num_spks, N, L)
            x = self.spe_block(x, input)
        
        x = x.view(B, self.num_spks, N, L)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        return x


class AudioDecoder(torch.nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # x: [B, N, L]
        if x.dim() not in [2, 3]: raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        x = torch.squeeze(x, dim=1) if torch.squeeze(x).dim() == 1 else torch.squeeze(x)
        return x
