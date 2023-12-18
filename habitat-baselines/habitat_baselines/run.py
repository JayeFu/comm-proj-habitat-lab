#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import random
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

if TYPE_CHECKING:
    from omegaconf import DictConfig

BASE_SAVE_PATH = "./train_dir/"


def build_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--resume_id",
        type=str,
        default=os.environ.get('JOB_UUID', None),
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()
    run_exp(args)


def execute_exp(config: "DictConfig", run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def run_exp(args) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval".
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(args.exp_config, args.opts)
    attempt_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    with read_write(config):
        # For logging
        config.habitat_baselines.wb.project_name = "comm-proj"
        config.habitat_baselines.wb.entity = "jiaweifu_ethz"
        if config.habitat_baselines.wb.run_name == "":
            config.habitat_baselines.wb.run_name = attempt_run_name
        config.habitat_baselines.wb.run_id = args.resume_id

        # For training specific
        if args.run_type == 'train':
            if not config.habitat.dataset.data_path.endswith('.json.gz'):
                config.habitat.dataset.data_path = os.path.join(config.habitat.dataset.data_path, "{split}/{split}.json.gz")

            config.habitat_baselines.tensorboard_dir = os.path.join(BASE_SAVE_PATH, args.resume_id, 'tb')
            config.habitat_baselines.checkpoint_folder = os.path.join(BASE_SAVE_PATH, args.resume_id, 'checkpoint')
            config.habitat_baselines.log_file = os.path.join(BASE_SAVE_PATH, args.resume_id, 'train.log')
        
        # For eval specific
        if args.run_type == 'eval':
            config.habitat_baselines.writer_type = 'tb'

            config.habitat_baselines.load_resume_state_config = False

    execute_exp(config, args.run_type)


if __name__ == "__main__":
    main()
