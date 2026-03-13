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

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform, RemoveChannelAugment


log = logging.getLogger(__name__)

test_cer = True

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig], ignored) -> Transform[Any, Any]:
        tfs = [instantiate(cfg) for cfg in configs]
        tfs.append(RemoveChannelAugment(ignored))
        return transforms.Compose(tfs)

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = module.load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
        )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
    )

    # Validate and test on the best checkpoint (if training), or on the
    # loaded `config.checkpoint` (otherwise)
    results = {}

    to_ignore = []
    # list of tuples (index, CER when index is removed)
    results = []
    for trial in range(10):
        print(f'Trial number {trial}')
        min_cer = None
        min_channel = None
        for i in range(16):
            ignored = []
            for e in to_ignore:
              ignored.append(e)

            if i in ignored:
              continue
            ignored.append(i)

            datamodule = instantiate(
                config.datamodule,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                train_sessions=_full_session_paths(config.dataset.train),
                val_sessions=_full_session_paths(config.dataset.val),
                test_sessions=_full_session_paths(config.dataset.test),
                train_transform=_build_transform(config.transforms.train, ignored),
                val_transform=_build_transform(config.transforms.val, ignored),
                test_transform=_build_transform(config.transforms.test, ignored),
                _convert_="object",
            )

            if test_cer:
                val_metrics = trainer.test(module, datamodule, verbose=False)
                cer = val_metrics[0]['test/CER']
            else:
                val_metrics = trainer.validate(module, datamodule, verbose=False)
                cer = val_metrics[0]['val/CER']
            
            if min_cer is None:
                min_cer = cer
                min_channel = i
            elif cer < min_cer:
                min_cer = cer
                min_channel = i
        
        results.append((min_channel, min_cer))
        to_ignore.append(min_channel)
    pprint.pprint(results)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()