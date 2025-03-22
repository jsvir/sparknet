from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed
from omegaconf import MISSING, ListConfig, OmegaConf

from nemo.collections.asr.parts.submodules.jasper import (
    JasperBlock,
    MaskedConv1d,
    SqueezeExcite,
    init_weights,
    jasper_activations,
)

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConvASREncoder']


class ConvASREncoder(NeuralModule, Exportable):
    """
    Convolutional encoder for ASR models. With this class you can implement JasperNet and QuartzNet models.

    Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
    """

    def _prepare_for_export(self, **kwargs):
        m_count = 0
        for name, m in self.named_modules():
            if isinstance(m, MaskedConv1d):
                m.use_mask = False
                m_count += 1

        Exportable._prepare_for_export(self, **kwargs)
        logging.warning(f"Turned off {m_count} masked convolutions")

    def input_example(self, max_batch=1, max_dim=8192):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        device = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=device)
        lens = torch.full(size=(input_example.shape[0],), fill_value=max_dim, device=device)
        return tuple([input_example, lens])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
        jasper,
        activation: str,
        feat_in: int,
        normalization_mode: str = "batch",
        residual_mode: str = "add",
        norm_groups: int = -1,
        conv_mask: bool = True,
        frame_splicing: int = 1,
        init_mode: Optional[str] = 'xavier_uniform',
        quantize: bool = False,
    ):
        super().__init__()
        if isinstance(jasper, ListConfig):
            jasper = OmegaConf.to_container(jasper)

        activation = jasper_activations[activation]()

        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        feat_in = feat_in * frame_splicing

        self._feat_in = feat_in

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', residual_mode)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_size', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            future_context = lcfg.get('future_context', -1)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    groups=groups,
                    separable=separable,
                    heads=heads,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation,
                    residual_panes=dense_res,
                    conv_mask=conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    se_context_window=se_context_window,
                    se_interpolation_mode=se_interpolation_mode,
                    kernel_size_factor=kernel_size_factor,
                    stride_last=stride_last,
                    future_context=future_context,
                    quantize=quantize,
                )
            )
            feat_in = lcfg['filters']

        self._feat_out = feat_in

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

        self.max_audio_length = 0

    def forward(self, audio_signal, length):
        self.update_max_sequence_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]
        return s_input[-1], length

    def update_max_sequence_length(self, seq_length: int, device):
        # Find global max audio length across all nodes
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            if seq_length < 5000:
                seq_length = seq_length * 2
            elif seq_length < 10000:
                seq_length = seq_length * 1.5
            self.max_audio_length = seq_length

            device = next(self.parameters()).device
            seq_range = torch.arange(0, self.max_audio_length, device=device)
            if hasattr(self, 'seq_range'):
                self.seq_range = seq_range
            else:
                self.register_buffer('seq_range', seq_range, persistent=False)

            # Update all submodules
            for name, m in self.named_modules():
                if isinstance(m, MaskedConv1d):
                    m.update_masked_length(self.max_audio_length, seq_range=self.seq_range)
                elif isinstance(m, SqueezeExcite):
                    m.set_max_len(self.max_audio_length, seq_range=self.seq_range)


@dataclass
class JasperEncoderConfig:
    filters: int = MISSING
    repeat: int = MISSING
    kernel: List[int] = MISSING
    stride: List[int] = MISSING
    dilation: List[int] = MISSING
    dropout: float = MISSING
    residual: bool = MISSING

    # Optional arguments
    groups: int = 1
    separable: bool = False
    heads: int = -1
    residual_mode: str = "add"
    residual_dense: bool = False
    se: bool = False
    se_reduction_ratio: int = 8
    se_context_size: int = -1
    se_interpolation_mode: str = 'nearest'
    kernel_size_factor: float = 1.0
    stride_last: bool = False


@dataclass
class ConvASREncoderConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASREncoder'
    jasper: Optional[JasperEncoderConfig] = field(default_factory=list)
    activation: str = MISSING
    feat_in: int = MISSING
    normalization_mode: str = "batch"
    residual_mode: str = "add"
    norm_groups: int = -1
    conv_mask: bool = True
    frame_splicing: int = 1
    init_mode: Optional[str] = "xavier_uniform"
