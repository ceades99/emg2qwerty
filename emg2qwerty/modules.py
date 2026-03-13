# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import math
from torch import nn


class AutoEncoder(nn.Module):
    """Attempt using 'Time Series Data Augmentation for Deep Learning: A Survey'
    Embeddings to generate synthetic data. Says do small transformations on
    embeddings to create better synthetic data than usual.
    """
    def __init__(
      self,
      input_size : int,
      hidden_in : int,
      hidden_out : int,
      output_size : int
    ):
      super().__init__()

      self.enc = nn.Sequential(
          nn.Conv1d(input_size, hidden_in, 5, 2, 2),
          nn.ReLU(),
          nn.Conv1d(hidden_in, hidden_out, 5, 2, 2),
          nn.ReLU(),
          nn.Conv1d(hidden_out, output_size, 5, 2, 2),
          nn.ReLU()
      )

      self.dec = nn.Sequential(
          nn.ConvTranspose1d(output_size, hidden_out, 5, 2, 2, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose1d(hidden_out, hidden_in, 5, 2, 2, output_padding=1),
          nn.ReLU(),
          nn.ConvTranspose1d(hidden_in, input_size, 5, 2, 2, output_padding=1)
      )
    
    def forward(self, inputs : torch.Tensor):
      outputs = self.enc(inputs)
      return self.dec(outputs)

class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        # Adding dropout as specified in the same paper after ReLU
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # dropout added
        x = self.dropout(x)

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        dropout: float = 0.3
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width, dropout),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class RnnLayer(nn.Module):
  def __init__(
    self,
    num_features: int,
    hidden : int,
    layers = 1,
    rnn_type = "RNN",
    bidirectional = False,
    dropout: float = 0.3
  ):
    super().__init__()

    # rnn doesnt work with dropout if layers ==1 
    if layers == 1:
      if rnn_type == "LSTM":
        self.rnn = nn.LSTM(num_features, hidden, layers, bidirectional=bidirectional)
      elif rnn_type == "GRU":
        self.rnn = nn.GRU(num_features, hidden, layers, bidirectional=bidirectional)
      else:
        self.rnn = nn.RNN(num_features, hidden, layers, bidirectional=bidirectional)
    
    else:
      if rnn_type == "LSTM":
        self.rnn = nn.LSTM(num_features, hidden, layers, bidirectional=bidirectional, dropout=dropout)
      elif rnn_type == "GRU":
        self.rnn = nn.GRU(num_features, hidden, layers, bidirectional=bidirectional, dropout=dropout)
      else:
        self.rnn = nn.RNN(num_features, hidden, layers, bidirectional=bidirectional, dropout=dropout)
  
  def forward(self, inputs: torch.Tensor):
    outputs, _ = self.rnn(inputs)
    return outputs

# Note: Test set has Dim 140,000 so seq_len needs to be
# very high in order to use this
class PositionalEncoding(nn.Module):
  def __init__(
    self,
    d_model: int,
    seq_len: int = 100
  ):
    super().__init__()

    self.d_model = d_model
    self.seq_len = seq_len

    pos_enc = torch.zeros(seq_len, 1, d_model)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pos_enc[:, 0, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 0, 1::2] = torch.cos(pos * div_term)

    self.register_buffer('pos_encoding', pos_enc)

  def forward(self, inputs : torch.Tensor):
    outputs = inputs + self.pos_encoding[:inputs.size(0)]
    return outputs

# Possible Solution to seq_len issue
class ConvPositionalEncoder(nn.Module):
  def __init__(
    self,
    d_model: int,
    kernel_size: int = 31
  ):
    super().__init__()

    self.pos = nn.Conv1d(
      d_model,
      d_model,
      kernel_size=kernel_size,
      stride=1,
      padding=kernel_size // 2,
      groups=d_model
    )
  
  def forward(self, inputs : torch.Tensor):
    output = self.pos(inputs.permute(1, 2, 0))
    return inputs + output.permute(2, 0, 1)

class TransformerLayer(nn.Module):
  def __init__(
    self,
    num_features: int,
    d_model: int = 256,
    kernel_size: int = 31,
    n_heads: int = 4,
    num_layers: int = 1
  ):
    super().__init__()

    self.linear = nn.Linear(num_features, d_model)
    self.pos_enc = ConvPositionalEncoder(d_model, kernel_size=kernel_size)

    transformers = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=n_heads,
      dim_feedforward=(d_model * 4),
      norm_first=True
    )

    self.tf = nn.TransformerEncoder(
      transformers,
      num_layers,
      norm=nn.LayerNorm(d_model)
    )

  def forward(self, inputs : torch.Tensor):
    outputs = self.linear(inputs)
    outputs = self.pos_enc(outputs)
    return self.tf(outputs)