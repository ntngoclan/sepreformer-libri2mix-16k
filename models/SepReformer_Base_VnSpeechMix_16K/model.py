import torch

from utils.decorators import *
from .modules.module import *


@logger_wraps()
class Model(torch.nn.Module):
    def __init__(self, 
                 num_stages: int, 
                 num_spks: int, 
                 module_audio_enc: dict, 
                 module_feature_projector: dict, 
                 module_separator: dict, 
                 module_output_layer: dict, 
                 module_audio_dec: dict):
        super().__init__()
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # Aux_loss
        self.out_layer_bn = torch.nn.ModuleList([])
        self.decoder_bn = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.out_layer_bn.append(OutputLayer(**module_output_layer, masking=True))
            self.decoder_bn.append(AudioDecoder(**module_audio_dec))
        
    def forward(self, x, return_aux=True, input_sizes=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.shape[-1] == 0:
            raise ValueError("Expected non-empty waveform [B, T] or [T].")
        original_length = x.shape[-1]
        if input_sizes is not None:
            lengths = torch.as_tensor(input_sizes, device=x.device)
            if (lengths.shape != (x.shape[0],) or torch.any(lengths != lengths.long())
                    or torch.any(lengths <= 0) or torch.any(lengths > original_length)):
                raise ValueError("Invalid waveform input_sizes.")
            lengths = lengths.long()
            # Only batch together equal true lengths. Padding is then confined
            # to the model's own stride/stage alignment, never another speaker
            # mixture's duration. Keep the fast path for full-length crops.
            if torch.any(lengths != original_length):
                grouped_indices, grouped_audio, grouped_aux = [], [], []
                for length in torch.unique(lengths).tolist():
                    indices = torch.nonzero(lengths == length, as_tuple=True)[0]
                    audio, auxiliary = self.forward(x[indices, :length], return_aux=return_aux)
                    grouped_indices.append(indices)
                    grouped_audio.append([torch.nn.functional.pad(t, (0, original_length-length)) for t in audio])
                    grouped_aux.append([[torch.nn.functional.pad(t, (0, original_length-length)) for t in stage]
                                        for stage in auxiliary])
                restore = torch.argsort(torch.cat(grouped_indices))
                audio = [torch.cat([group[j] for group in grouped_audio])[restore] for j in range(self.num_spks)]
                auxiliary = [[torch.cat([group[r][j] for group in grouped_aux])[restore]
                              for j in range(self.num_spks)] for r in range(self.num_stages)] if return_aux else []
                return audio, auxiliary
        stride = self.audio_encoder.conv1d.stride[0]
        kernel = self.audio_encoder.conv1d.kernel_size[0]
        # BatchNorm in training needs at least two values at the bottleneck
        # when a short utterance forms a single-item length group.
        minimum = kernel + stride * (2 ** (self.num_stages + 1) - 1) if self.training else kernel
        padded_length = max(minimum, (original_length + stride - 1) // stride * stride)
        x = torch.nn.functional.pad(x, (0, padded_length - original_length))
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(each_spk_output[idx])[..., :original_length] for idx in range(self.num_spks)]
        if not return_aux:
            return audio, []
        
        # Aux_loss
        audio_aux = []
        for idx, each_stage_output in enumerate(each_stage_outputs):
            each_stage_output = self.out_layer_bn[idx](
                torch.nn.functional.interpolate(
                    each_stage_output, size=encoder_output.shape[-1], mode="nearest"
                ),
                encoder_output,
            )
            out_aux = [each_stage_output[jdx] for jdx in range(self.num_spks)]
            audio_aux.append([self.decoder_bn[idx](out_aux[jdx])[..., :original_length] for jdx in range(self.num_spks)])
            
        return audio, audio_aux
