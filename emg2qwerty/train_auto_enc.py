# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from emg2qwerty.lightning import WindowedEMGDataModule, AutoEncoderModule

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform, Compose, ToTensor, Lambda


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    to_tensor = Compose([
        ToTensor(fields=("emg_left", "emg_right"), stack_dim=-1),
        Lambda(lambda x : x.view(x.shape[0], -1))
    ])

    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    datamodule = WindowedEMGDataModule(
        window_length=1000,
        padding=(0, 0),
        batch_size=32,
        num_workers=4,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=to_tensor,
        val_transform=to_tensor,
        test_transform=to_tensor
    )

    model = AutoEncoderModule(
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        input_size=16 * 2,
        hidden_in=64,
        hidden_out=128,
        output_size=32,
        lr=1e-3
    )

    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()