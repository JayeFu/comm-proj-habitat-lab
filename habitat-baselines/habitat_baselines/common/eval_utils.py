import shutil
import json
import os
import subprocess
import time
from typing import Any, Dict, List, TYPE_CHECKING, Union

import imageio.v3 as iio
import torch

from habitat import logger
from habitat.utils.visualizations.utils import (
    observations_to_image,
    images_to_video,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    WeightsAndBiasesWriter,
)
from habitat_baselines.common.tensor_dict import (
    TensorDict,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


import numpy as np
import wandb


class EvaluationMonitor:
    def __init__(
        self,
        config: "DictConfig",
    ):
        self._config = config
        self._train_save_dir = os.path.dirname(self._config.habitat_baselines.log_file)
        self._train_cfg_fp = os.path.join(
            self._train_save_dir,
            "exp_cfg.yaml"
        )

        self._ckpt_path_tpl = os.path.join(
            self._train_save_dir,
            'checkpoint',
            'ckpt.{ckpt_idx}.pth'
        )
        self._eval_tb_dir_tpl = os.path.join(
            self._train_save_dir,
            'tb_eval_{ckpt_idx}'
        )
        self._eval_log_fp_tpl = os.path.join(
            self._train_save_dir,
            'eval_ckpt.{ckpt_idx}.log'
        )
        self._eval__stat_log_fp_tpl = os.path.join(
            self._train_save_dir,
            'eval_ckpt.{ckpt_idx}.stat.log'
        )
        self._eval_video_dir_tpl = os.path.join(
            self._train_save_dir,
            'vid_eval_{ckpt_idx}'
        )

        self._evaled_ckpt_idxs: List[int] = []
        self._processed_ckpt_idxs: List[int] = []
        self._ckpt_idx_steps: Dict[int, int] = {}

    def launch_evaluation(
        self,
        ckpt_idx: int,
        step: int
    ):
        ckpt_path = self._ckpt_path_tpl.format(ckpt_idx=ckpt_idx)
        eval_tb_dir = self._eval_tb_dir_tpl.format(ckpt_idx=ckpt_idx)
        eval_log_file = self._eval_log_fp_tpl.format(ckpt_idx=ckpt_idx)

        logger.info(f"Launch eval for ckpt {ckpt_idx}")
        subprocess.Popen(
            [
                "/usr/bin/bash",
                "./launch_eval.sh",
                self._train_cfg_fp,  # $1, EXP_CFG
                str(ckpt_idx),  # $2 CKPT_NAME
                ckpt_path,  # $3, EVAL_CKPT_PATH_DIR
                eval_tb_dir,  # $4, TENSORBOARD_DIR
                eval_log_file,  # $5, LOG_FILE
            ],
            stdout=subprocess.DEVNULL,
        )

        self._evaled_ckpt_idxs.append(ckpt_idx)
        self._ckpt_idx_steps[ckpt_idx] = step

    def monitor_stat_logs(self, writer: Union[TensorboardWriter, WeightsAndBiasesWriter], accumulated_logs: Dict[int, Dict]):
        if len(self._processed_ckpt_idxs) >= len(self._evaled_ckpt_idxs):  # No pending eval ckpt, just log
            for log_step in accumulated_logs.copy().keys():
                writer.add_aggregated_logs(
                        accumulated_logs[log_step],
                        log_step
                    )

                accumulated_logs.pop(log_step)

            return

        for ckpt_idx in self._evaled_ckpt_idxs:
            if ckpt_idx in self._processed_ckpt_idxs:
                continue

            eval_stat_log_fp = self._eval__stat_log_fp_tpl.format(ckpt_idx=ckpt_idx)
            if not os.path.exists(eval_stat_log_fp):  # log not created
                continue

            if not self._is_write_finished(fp=eval_stat_log_fp):  # Still writing
                continue

            logger.info(f"Log {eval_stat_log_fp} into writer")
            with open(eval_stat_log_fp, 'r') as f:
                eval_stats = json.load(f)

            step = eval_stats['step']
            aggregated_stats = eval_stats['aggregated_stats']

            step_stats = {"eval_reward/average_reward": aggregated_stats["reward"]}
            metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
            for k, v in metrics.items():
                step_stats[f"eval_metrics/{k}"] = v

            eval_video_dir = self._eval_video_dir_tpl.format(ckpt_idx=ckpt_idx)
            for video_name_full in os.listdir(eval_video_dir):
                video_name = video_name_full.split('.')[0]
                frames = iio.imread(os.path.join(eval_video_dir, video_name_full))  # np.ndarray: (N, H, W, 3)
                frames = np.transpose(frames, axes=(0, 3, 1, 2))  # (N, H, W, 3) -> (N, 3, H, W)
                step_stats[f"eval_video/{video_name}"] = wandb.Video(
                    frames,
                    fps=self._config.habitat_baselines.video_fps,
                )

            accumulated_logs[step].update(step_stats)

            def _is_lower(val, upper):
                if upper is None:
                    return True
                return val < upper

            upper_step = None
            if ckpt_idx + 1 in self._evaled_ckpt_idxs:  # ckpt_idx + 1 also evaled
                upper_step = self._ckpt_idx_steps[ckpt_idx + 1]

            for log_step in accumulated_logs.copy().keys():
                if log_step >= step and _is_lower(log_step, upper_step):
                    writer.add_aggregated_logs(
                        accumulated_logs[log_step],
                        log_step
                    )

                    accumulated_logs.pop(log_step)

            self._processed_ckpt_idxs.append(ckpt_idx)

    @staticmethod
    def _is_write_finished(fp: str, est_write_time: float = 1.0):
        cur_t = time.time()
        mod_t = os.path.getmtime(fp)

        return (cur_t - mod_t) > est_write_time


class EvaluationSaver:
    def __init__(
        self,
        save_episode_id: int,
        eval_log_fp: str,
        disable_logging: bool = False,
    ) -> None:
        self._disable_logging = disable_logging

        self._train_save_dir = os.path.dirname(eval_log_fp)
        ckpt_idx = eval_log_fp.split('/')[-1].split('.')[1]  # 0: eval_ckpt, 2: log
        self.save_dir = os.path.join(
            self._train_save_dir,
            f"vid_eval_{ckpt_idx}"
        )
        self.video_name = f"episode={save_episode_id}"

        self._save_episode_id = save_episode_id
        self._rgb_frames = []  # type: List[np.ndarray]

    def add_observations(
        self, 
        episode_id: int, 
        env_idx: int,
        batch: TensorDict, 
        not_done_masks: torch.Tensor,
        infos: Dict[str, Any]
    ):
        if episode_id == self._save_episode_id:  # only save this episode
            frame = observations_to_image(
                {k: v[env_idx] for k, v in batch.items()}, infos[env_idx]
            )
            if not not_done_masks[env_idx].item():
                # The last frame corresponds to the first frame of the next episode
                # but the info is correct. So we use a black frame
                frame = observations_to_image(
                    {k: v[env_idx] * 0.0 for k, v in batch.items()}, infos[env_idx]
                )

            self._rgb_frames.append(frame)

    def save_video(
        self,
        episode_id: int,
        video_option: List[str],
        fps: int = 10,
    ):
        if episode_id == self._save_episode_id:
            if not self._disable_logging and "wandb" in video_option:
                images_to_video(
                    images=self._rgb_frames,
                    output_dir=self.save_dir,
                    video_name=self.video_name,
                    fps=fps,
                    verbose=False
                )

            self._rgb_frames = []  # type: List[np.ndarray]

    def write_eval_stats_to_disk(
        self,
        aggregated_stats: Dict[str, float],
        step: int,
        update: int,
        eval_log_fp: str
    ):
        if self._disable_logging:  # Don't write
            return

        stat_log_fp = eval_log_fp[:-len('.log')] + '.stat.log'
        eval_stats = dict(
            step=step,
            update=update,
            aggregated_stats=aggregated_stats
        )

        with open(stat_log_fp, 'w') as f:
            json.dump(eval_stats, f)

    def delete_saved_files(
        self,
    ):
        logger.info(f"Delete saved files in {self._train_save_dir}")
        shutil.rmtree(self._train_save_dir, ignore_errors=True)
