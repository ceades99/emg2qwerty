# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio
import random


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

@dataclass
class IntensityAugment:
    intensity_mod : float = 0.2 # maximum modification value fr * (1.0 plus or minus this)
    pr : float = 0.3 # probability a frequency is modified

    def __call__(self, specgram : torch.Tensor):
        if random.random() < self.pr:
            scale = random.uniform(1.0 - self.intensity_mod, 1.0 + self.intensity_mod)
            specgram = specgram * scale
        return specgram

@dataclass
class ReversedElectrodesAugment:
    """A data augmentation techinque suggested in https://ieeexplore.ieee.org/document/8664790
    where all electrodes are placed backwards (Condition 1-2).
    """
    pr : float = 0.3

    def __call__(self, specgram : torch.Tensor):
        if random.random() < self.pr:
            # I think since channels is 2nd to last I flip it along channels
            return torch.flip(specgram, dims=[-2])
        return specgram

@dataclass
class RandomSwapAugment:
    """A data augmentation technique suggested in the same paper where
    two random channels are swapped (Condition 2)
    """
    pr : float = 0.3

    def __call__(self, specgram : torch.Tensor):
        if random.random() < self.pr:
            num_channels = specgram.shape[-2]
            channel1 = random.randint(0, num_channels - 1)
            channel2 = random.randint(0, num_channels - 1)

            c1_vals = specgram[..., channel1, :].clone()
            # channel2 stored in channel1
            specgram[..., channel1, :] = specgram[..., channel2, :]
            # channel1 copy stored in channel2
            specgram[..., channel2, :] = c1_vals
        return specgram

@dataclass
class UserErrorAugment:
  """This class combines the previous ones to choose a single error augmentation.
  It seems rather unlikely that the RandomSwap and the 
  Reversed electrodes would occur in the same data collection session, so
  this will only pick 1.
  """
  pr : float = 0.3

  def __call__(self, specgram : torch.Tensor):
      if random.random() < self.pr:
          if random.random() < 0.5:
              # do swap
              return torch.flip(specgram, dims=[-2])
          else:
              num_channels = specgram.shape[-2]
              channel1 = random.randint(0, num_channels - 1)
              channel2 = random.randint(0, num_channels - 1)

              c1_vals = specgram[..., channel1, :].clone()
              # channel2 stored in channel1
              specgram[..., channel1, :] = specgram[..., channel2, :]
              # channel1 copy stored in channel2
              specgram[..., channel2, :] = c1_vals

              return specgram
      return specgram

@dataclass
class BadSensorAugment:
    pr : float = 0.05
    
    def __call__(self, specgram : torch.Tensor):
        processed = specgram.clone()

        # iterate over each channel
        for channel in range(specgram.shape[-2]):
            # zero ouit a channel with probability pr
            if random.random() < self.pr:
                processed[..., channel, :] = 0.0
        

        return processed

@dataclass
class RemoveChannelAugment:
    """Removes a given channel. Meant to be used during validation runs in
    order to see if accuracy is preserved when certain channels are removed.
    """
    channels: Sequence[int] = ()

    def __call__(self, specgram : torch.Tensor):
        if not self.channels:
            return specgram

        processed = specgram.clone()
        for channel in self.channels:
            processed[..., channel, :] = 0.0
        return processed

@dataclass
class GaussianNoise:
    noise: float = 1.0
    pr: float = 0.4

    def __call__(self, specgram : torch.Tensor):
        if random.random() < self.pr:
            noise = torch.randn_like(specgram) * self.noise
            return specgram + noise
        return specgram

@dataclass
class DecreaseSampleRate:
    """Takes every nth time step using `self.rate`.
    """
    rate : int = 1

    def __call__(self, specgram : torch.Tensor):
        return specgram[::self.rate, ...]