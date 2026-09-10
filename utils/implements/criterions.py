import torch
import numpy as np

from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale
from dataclasses import dataclass, field, fields
from typing import List, Type, Any, Callable, Optional, Union
from loguru import logger
from utils.decorators import *
from mir_eval.separation import bss_eval_sources


# Utility functions
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)


def length_mask(reference, input_sizes):
    """Build a [B, T] mask so padded samples do not affect PIT objectives."""
    lengths = input_sizes.to(device=reference.device, dtype=torch.long)
    lengths = lengths.clamp(min=1, max=reference.shape[-1])
    sample_index = torch.arange(reference.shape[-1], device=reference.device)
    return (sample_index.unsqueeze(0) < lengths.unsqueeze(1)).to(reference.dtype)


def masked_zero_mean(signal, mask):
    valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (signal * mask).sum(dim=-1, keepdim=True) / valid_count
    return (signal - mean) * mask

@dataclass(slots=True)
class STFTBase(torch.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Uncentered frames; at least one frame for short audio.
    """
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    K: torch.nn.Parameter = field(init=False)
    num_bins: int = field(init=False)

    def __post_init__(self):
        super(STFTBase, self).__init__()  # Initialize the torch.nn.Module base class
        K = self._init_kernel(self.frame_length, self.frame_shift)
        self.K = torch.nn.Parameter(K, requires_grad=False).to(self.device)
        self.num_bins = self.K.shape[0] // 2
    
    def _init_kernel(self, frame_len, frame_hop):
        # FFT points
        N = frame_len
        # window
        if self.window == 'hann':
            W = torch.hann_window(frame_len)
        if N//4 == frame_hop:
            const = (2/3)**0.5       
            W = const*W
        elif N//2 == frame_hop:
            W = W**0.5
        S = 0.5 * (N * N / frame_hop)**0.5
        
        # Updated FFT calculation for efficiency
        K = torch.fft.rfft(torch.eye(N) / S, dim=1)[:frame_len]
        K = torch.stack((torch.real(K), torch.imag(K)), dim=2)
        K = torch.transpose(K, 0, 2) * W # 2 x N/2+1 x F
        K = torch.reshape(K, (N + 2, 1, frame_len)) # N+2 x 1 x F
        return K

    def extra_repr(self):
        return (f"window={self.window}, stride={self.frame_shift}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}")

@logger_wraps()
@dataclass(slots=True)
class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        if x.shape[-1] == 0:
            raise ValueError("STFT requires non-empty audio.")
        # Preserve the original uncentered frame convention, with one frame
        # minimum for short inputs. The loss masks frames by each true length.
        len_padded = max(self.frame_length, ceil(x.shape[-1] / self.frame_shift) * self.frame_shift)
        x = torch.nn.functional.pad(x, (0, len_padded - x.shape[-1]))
        multi_channel = x.dim() == 3
        if multi_channel:
            batch, channels, samples = x.shape
            x = x.reshape(batch * channels, 1, samples)
        else:
            x = x.unsqueeze(1)
        c = torch.nn.functional.conv1d(x, self.K, stride=self.frame_shift)
        if multi_channel:
            c = c.reshape(batch, channels, -1, c.shape[-1])
        r, i = torch.chunk(c, 2, dim=2 if multi_channel else 1)

        if cplx:
            return r, i
        m = (r**2 + i**2 + 1.0e-10)**0.5
        p = torch.atan2(i, r)
        return m, p

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNR_mag:
    device: torch.device
    frame_length: int
    frame_shift: int
    window: str
    num_stages: int
    num_spks: int
    scale_inv: bool
    mel_opt: bool
    
    
    stft: List[Any] = field(init=False)
    mel_fb: Callable[[torch.Tensor], torch.Tensor] = field(init=False)
    
    def __post_init__(self):
        self.stft = [STFT(self.device, self.frame_length, self.frame_shift, self.window) for _ in range(self.num_stages)]
        self.mel_fb = MelScale(n_mels=80, sample_rate=16000, n_stft=int(self.frame_length / 2) + 1).to(self.device) if self.mel_opt else lambda x: x

    def __repr__(self):
        # __init__
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]

        # __post_init__
        stft_repr = f"stft = [STFT instance for {len(self.stft)} layers]"
        mel_fb_repr = "mel_fb = MelScale" if self.mel_opt else "mel_fb=Identity"
        post_init_reprs = [stft_repr, mel_fb_repr]

        return f"<{class_name}({', '.join(field_strs + post_init_reprs)})>"
    
    def _center(self, signals, input_sizes):
        signals = torch.stack([value.to(self.device) for value in signals], dim=1)
        lengths = input_sizes.to(device=self.device)
        if (lengths.shape != (signals.shape[0],) or torch.any(lengths != lengths.long())
                or torch.any(lengths <= 0) or torch.any(lengths > signals.shape[-1])):
            raise ValueError("Invalid true lengths for spectral PIT.")
        lengths = lengths.long()
        mask = length_mask(signals[:, 0], lengths).unsqueeze(1)
        return masked_zero_mean(signals, mask), lengths

    def prepare_targets(self, targets, input_sizes):
        """Batch-local reference cache; reusable across all auxiliary heads."""
        centered, lengths = self._center(targets, input_sizes)
        r, i = self.stft[0](centered, cplx=True)
        return centered, r.square() + i.square(), lengths

    def __call__(self, **kwargs):
        eps = 1.0e-12
        estimates, lengths = self._center(kwargs['estims'], kwargs['input_sizes'])
        prepared = kwargs.get('prepared_targets')
        if prepared is None:
            prepared = self.prepare_targets(kwargs['target_attr'], kwargs['input_sizes'])
        targets, target_power, target_lengths = prepared
        if estimates.shape != targets.shape or not torch.equal(lengths, target_lengths):
            raise ValueError("Spectral target cache does not match this batch.")
        r, i = self.stft[kwargs['idx']](estimates, cplx=True)
        estimate_mag = (r.square() + i.square() + 1e-10).sqrt()
        # [B, estimate speaker, reference speaker]. Scale BEFORE adding the
        # magnitude epsilon, exactly as waveform scaling before linear STFT.
        if self.scale_inv:
            scale = torch.einsum('bet,brt->ber', estimates, targets)
            scale = (scale / (targets.square().sum(-1).unsqueeze(1) + eps)).clamp_min(1e-2)
        else:
            scale = estimates.new_ones(estimates.shape[0], self.num_spks, self.num_spks)
        reference_mag = (target_power.unsqueeze(1) * scale[..., None, None].square() + 1e-10).sqrt()
        if self.mel_opt:
            estimate_mag = self.mel_fb(estimate_mag)
            reference_mag = self.mel_fb(reference_mag)
        padded_lengths = ((lengths + self.frame_shift - 1) // self.frame_shift * self.frame_shift).clamp_min(self.frame_length)
        frame_counts = (padded_lengths - self.frame_length) // self.frame_shift + 1
        frame_mask = (torch.arange(estimate_mag.shape[-1], device=self.device)[None, :] < frame_counts[:, None])
        frame_mask = frame_mask[:, None, None, None, :]
        reference_mag = reference_mag * frame_mask
        error = (estimate_mag.unsqueeze(2) * frame_mask) - reference_mag
        cost = -20 * torch.log10(eps + torch.linalg.vector_norm(reference_mag, dim=(-2, -1))
                                / (torch.linalg.vector_norm(error, dim=(-2, -1)) + eps))
        scores = torch.stack([sum(cost[:, speaker, target] for speaker, target in enumerate(perm))
                              for perm in permutations(range(self.num_spks))])
        return scores.min(dim=0).values.mean()

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNR_time:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"
    
    def __call__(self, **kwargs):
        estims = kwargs['estims']
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [target.to(self.device) for target in kwargs["target_attr"]]
        
        def _SDR_loss(permute, eps=1.0e-8):
            loss_for_permute = []
            for s, t in enumerate(permute):
                mix = estims[s]
                src = targets[t]
                mask = length_mask(mix, input_sizes)
                mix_zm = masked_zero_mean(mix, mask)
                src_zm = masked_zero_mean(src, mask)
                src_zm_scale = src_zm
                if self.scale_inv:
                    scale_factor = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
                    src_zm_scale = scale_factor * src_zm
                
                utt_loss = - 20 * torch.log10(eps + l2norm(src_zm_scale) / (l2norm(mix_zm - src_zm_scale) + eps))
                utt_loss = torch.clamp(utt_loss, min=-30)
                
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)
        
        pscore = torch.stack([_SDR_loss(p) for p in permutations(range(self.num_spks))])
        min_perutt, _ = torch.min(pscore, dim=0)
        num_utts = input_sizes.shape[0]
        return torch.sum(min_perutt) / num_utts

@logger_wraps()
@dataclass(slots=True)
class PIT_SISNRi:
    device: torch.device
    num_spks: int
    scale_inv: bool

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"
    
    def __call__(self, **kwargs):
        estims = kwargs['estims']
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        input = kwargs['mixture'].to(self.device)
        input_mask = length_mask(input, input_sizes)
        input_zm = masked_zero_mean(input, input_mask)
        eps = kwargs['eps']
        
        def _SDR_loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                est = estims[s]
                src = targets[t]
                mask = length_mask(est, input_sizes)
                est_zm = masked_zero_mean(est, mask)
                src_zm = masked_zero_mean(src, mask)
                src_zm_s = src_zm
                if self.scale_inv:
                    src_zm_s = torch.sum(est_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps) * src_zm
                
                utt_loss_est = 20 * torch.log10(eps + l2norm(src_zm_s) / (l2norm(est_zm - src_zm_s) + eps))
                src_zm_x = src_zm
                if self.scale_inv:
                    src_zm_x = torch.sum(input_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps) * src_zm
                utt_loss_in = 20 * torch.log10(eps + l2norm(src_zm_x) / (l2norm(input_zm - src_zm_x) + eps))
                loss_for_permute.append(utt_loss_est - utt_loss_in)
            return torch.stack(loss_for_permute)
        
        # [num_permutations, num_speakers, batch]. Select one permutation per
        # utterance using the sum across speakers, then return its per-speaker
        # scores for CSV reporting.
        pscore = torch.stack(
            [_SDR_loss(p) for p in permutations(range(self.num_spks))], dim=0
        )
        best_perutt, best_perm = torch.max(pscore.sum(dim=1), dim=0)
        batch_index = torch.arange(pscore.shape[-1], device=pscore.device)
        best_per_source = pscore.permute(2, 0, 1)[batch_index, best_perm]
        num_utts = input_sizes.shape[0]
        return torch.sum(best_perutt) / num_utts, best_per_source.transpose(0, 1)

@logger_wraps()
@dataclass(slots=True)
class PIT_SDRi:
    device: torch.device
    dump: int

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [f"{field.name}={getattr(self, field.name)!r}" for field in init_fields]
        return f"<{class_name}({', '.join(field_strs)})>"
    
    def __call__(self, **kwargs):
        estims = torch.stack(kwargs['estims'], dim=0).squeeze(1)
        input_sizes = kwargs["input_sizes"].to(self.device)
        targets = [t.to(self.device) for t in kwargs["target_attr"]]
        targets = torch.stack(targets, dim=0).squeeze(1)
        input = torch.cat([kwargs['mixture'], kwargs['mixture']], dim=0)
        
        targets = targets.cpu().data.numpy()
        estims = estims.cpu().data.numpy()
        input = input.cpu().data.numpy()

        min_perutt_out, _, _, _ = bss_eval_sources(targets, estims)
        min_perutt_in, _, _, _ = bss_eval_sources(targets, input)
        
        num_utts = input_sizes.shape[0]
        return np.sum(min_perutt_out - min_perutt_in) / num_utts, min_perutt_out - min_perutt_in
