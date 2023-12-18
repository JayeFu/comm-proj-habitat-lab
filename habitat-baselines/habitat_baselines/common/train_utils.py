from typing import Any, Dict, List, TYPE_CHECKING, Union

import os
import json
import numpy as np
import pandas as pd
import subprocess
import time
import wandb

import imageio.v3 as iio
from habitat import logger
from habitat_baselines.common.tensorboard_utils import WeightsAndBiasesWriter
from habitat_baselines.common.tensor_dict import (
    TensorDict,
)
from habitat.utils.visualizations.utils import observations_to_image

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TrainVideoBuffer:
    def __init__(
        self,
        num_envs: int,
        video_fps: int,
        disable_logging: bool = False,
    ) -> None:
        self._disable_logging = disable_logging

        self._num_envs = num_envs
        self._rgb_frames = [
            [] for _ in range(self._num_envs)
        ]  # type: List[List[np.ndarray]]

        self._video_fps = video_fps

    def add_observations(
        self, 
        batch: TensorDict, 
        infos: Dict[str, Any], 
        dones: List[bool],
    ) -> None:
        if self._disable_logging:  # don't cache if logging disabled, it takes time
            return

        for env_idx, env_frames in enumerate(self._rgb_frames):
            frame = observations_to_image(
                {k: v[env_idx] for k, v in batch.items()}, infos[env_idx]
            )
            # Done. The last frame corresponding to the first frame of the next episode
            # but the info is correct. So we use a black frame
            if dones[env_idx]:
                frame = observations_to_image(
                    {k: v[env_idx] * 0.0 for k, v in batch.items()}, infos[env_idx]
                )
                env_frames.append(frame)

            env_frames.append(frame)

    def retrieve_frames(
        self, 
        env_idxs: List[int] = [-1],
    ) -> Dict[str, wandb.Video]:
        if self._disable_logging:  # don't retrieve, nothing inside
            return {}

        if -1 in env_idxs:  # retrieve all
            env_idxs = [i for i in range(self._num_envs)]

        video_log = {}
        for i in env_idxs:
            frames = self._rgb_frames[i]
            if len(frames) <= 0:
                continue
            # initial shape of np.ndarray list: N * (H, W, 3)
            frames = [
                np.transpose(np_arr, axes=(2, 0, 1))  # (H, W, 3) -> (3, H, W)
                for np_arr in frames
            ]  # N * (3, H, W)
            frames = np.stack(frames, axis=0)
            # final shape of np.ndarray: (N, 3, H, W)

            video_log[f"rollout/env{i}"] = wandb.Video(frames, fps=self._video_fps)

        return video_log
    
    def on_rollout_end(self) -> None:
        self._rgb_frames = [
            [] for _ in range(self._num_envs)
        ]  # type: List[List[np.ndarray]]


class EvalMonitor:
    def __init__(
        self,
        train_log_file: str,
        video_fps: int,
        disable_logging: bool = False
    ) -> None:
        self._disable_logging = disable_logging

        self._setup_path_templates(
            train_log_file=train_log_file
        )

        self._evaled_ckpt_idxs: List[int] = []
        self._logged_ckpt_idxs: List[int] = []

        self._eval_stats_l: List[Dict] = []
        self._eval_videos_l: List[Dict] = []

        self._video_fps = video_fps

    def _setup_path_templates(self, train_log_file: str):
        self._train_save_dir = os.path.dirname(train_log_file)

        self._train_cfg_fp = os.path.join(
            self._train_save_dir,
            "exp_cfg.yaml"
        )
        self._ckpt_save_dir = os.path.join(
            self._train_save_dir,
            'checkpoint'
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
        self._eval_stat_log_fp_tpl = os.path.join(
            self._train_save_dir,
            'eval_ckpt.{ckpt_idx}.stat.log'
        )
        self._eval_video_dir_tpl = os.path.join(
            self._train_save_dir,
            'vid_eval_{ckpt_idx}'
        )

    def launch_evaluation_during_training(
        self,
        ckpt_idx: int
    ):
        self.launch_evaluation_for(ckpt_idx=ckpt_idx, block=False)
        self._evaled_ckpt_idxs.append(ckpt_idx)

    def retrieve_eval_results_during_training(self):
        if self._disable_logging:  # don't read, no results written
            return {}

        if not self.has_pending_eval_results():
            # No pending ckpt stats to log
            return {}
        
        next_ckpt_idx = -1
        for ckpt_idx in self._evaled_ckpt_idxs:
            if ckpt_idx not in self._logged_ckpt_idxs:
                next_ckpt_idx = ckpt_idx

                # We just take care of the next one pending ckpt idx
                break

        eval_stat_log_fp = self._eval_stat_log_fp_tpl.format(ckpt_idx=ckpt_idx)
        if not os.path.exists(eval_stat_log_fp):
            # Evaluation not finished yet
            return {}

        eval_stats, eval_videos = self.read_eval_results_for(ckpt_idx=next_ckpt_idx)

        aggregated_stats = eval_stats.copy()
        aggregated_stats.update(eval_videos)

        self._logged_ckpt_idxs.append(next_ckpt_idx)
        
        return aggregated_stats
    
    def has_pending_eval_results(self):
        return len(self._logged_ckpt_idxs) < len(self._evaled_ckpt_idxs)

    def launch_evaluation_after_training(
        self,
        writer: WeightsAndBiasesWriter,
    ):
        for ckpt_fn in os.listdir(self._ckpt_save_dir):
            if not 'ckpt' in ckpt_fn:  # Do not evaluate latest.pth
                continue

            ckpt_idx = ckpt_fn.split('.')[1]  # 0: ckpt, 2: pth

            self.launch_evaluation_for(ckpt_idx=ckpt_idx, block=True)

            if not self._disable_logging:  # read if logging enabled, i.e., eval results written
                eval_stats, eval_videos = self.read_eval_results_for(ckpt_idx=ckpt_idx)
                self._eval_stats_l.append(eval_stats)
                self._eval_videos_l.append(eval_videos)

        if not self._disable_logging:  # log if logging enabled
            self.log_all_eval_results(writer=writer)

    def launch_evaluation_for(
        self,
        ckpt_idx: Union[str, int],
        block: bool = True,
    ):
        ckpt_path = self._ckpt_path_tpl.format(ckpt_idx=ckpt_idx)
        eval_tb_dir = self._eval_tb_dir_tpl.format(ckpt_idx=ckpt_idx)
        eval_log_file = self._eval_log_fp_tpl.format(ckpt_idx=ckpt_idx)

        logger.info(f"Launch eval for ckpt {ckpt_idx}")
        if block:  # blocking subprocess
            launch_func = subprocess.run
        else:  # non-blocking subprocess
            launch_func = subprocess.Popen
        launch_func(
            [
                "/usr/bin/bash",
                "./launch_eval.sh",
                self._train_cfg_fp,  # $1, EXP_CFG
                str(ckpt_idx),  # $2 CKPT_NAME
                ckpt_path,  # $3, EVAL_CKPT_PATH_DIR
                eval_tb_dir,  # $4, TENSORBOARD_DIR
                eval_log_file,  # $5, LOG_FILE
            ],
            stdout=subprocess.DEVNULL
        )

    def read_eval_results_for(
        self,
        ckpt_idx: Union[str, int],
    ):
        logger.info(f"Read eval results for ckpt {ckpt_idx}")

        eval_stat_log_fp = self._eval_stat_log_fp_tpl.format(ckpt_idx=ckpt_idx)
        with open(eval_stat_log_fp, 'r') as f:
            eval_results = json.load(f)

        step_id = eval_results['step']
        update_id = eval_results['update']
        aggregated_stats = eval_results['aggregated_stats']

        eval_stats = {
            "eval/ckpt_idx": int(ckpt_idx),
            "eval/update": update_id,
            "eval/step": step_id,
            "eval/average_reward": aggregated_stats["reward"]
        }
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            eval_stats[f"eval/{k}"] = v

        # self._eval_stats_l.append(eval_stats)

        logger.info(f"Read eval videos for ckpt {ckpt_idx}")

        eval_videos = {"eval/update": update_id}
        eval_video_dir = self._eval_video_dir_tpl.format(ckpt_idx=ckpt_idx)
        for video_name_full in os.listdir(eval_video_dir):
            video_name = video_name_full.split('.')[0]
            frames = iio.imread(os.path.join(eval_video_dir, video_name_full))  # np.ndarray: (N, H, W, 3)
            frames = np.transpose(frames, axes=(0, 3, 1, 2))  # (N, H, W, 3) -> (N, 3, H, W)

            eval_videos[f"eval_video/{video_name}"] = wandb.Video(
                frames,
                fps=self._video_fps,
            )

        return eval_stats, eval_videos

    def log_all_eval_results(
        self, 
        writer: WeightsAndBiasesWriter,
        sleep_sec: float = 1.0,
    ):
        logger.info(f"Log eval results for {self._train_save_dir}")

        # Sort
        self._eval_stats_l.sort(key=lambda x: x["eval/update"])
        self._eval_videos_l.sort(key=lambda x: x["eval/update"])

        # Log eval stats into artifact
        # eval_stats_df = pd.DataFrame(self._eval_stats_l).set_index('eval/ckpt_idx')
        # writer.add_eval_stats(
        #     eval_stats_df=eval_stats_df
        # )
        # time.sleep(sleep_sec)

        # Log aggregated stats
        for eval_stats, eval_videos in zip(self._eval_stats_l, self._eval_videos_l):
            aggregated_stats = eval_stats.copy()
            aggregated_stats.update(eval_videos)

            writer.add_aggregated_logs(
                aggregated_stats,
                step_id=None  # just commit
            )
            time.sleep(sleep_sec)


class TrainUtils:
    def __init__(
        self,
        num_envs: int,
        config: "DictConfig",
        disable_logging: bool = False
    ) -> None:
        self._disable_logging = disable_logging

        self._train_vid_buf = TrainVideoBuffer(
            num_envs=num_envs,
            video_fps=config.habitat_baselines.video_fps,
            disable_logging=self._disable_logging
        )
        self._video_interval = config.habitat_baselines.video_interval

        self._eval_monitor = EvalMonitor(
            train_log_file=config.habitat_baselines.log_file,
            video_fps=config.habitat_baselines.video_fps,
            disable_logging=self._disable_logging
        )

    def cache_train_videos(
        self, 
        batch: TensorDict, 
        infos: Dict[str, Any], 
        dones: List[bool], 
        num_updates_done: int
    ):
        # only add frames if need to retrieve videos
        # since observations_to_image is slow
        # Need to plus 1 since it will increase 1 after agent update
        if not self._disable_logging and (num_updates_done + 1) % self._video_interval == 0:
            self._train_vid_buf.add_observations(
                batch=batch,
                infos=infos,
                dones=dones
            )

    def retrieve_train_videos(
        self, 
        env_idxs: List[int] = [-1],
    ):
        if self._disable_logging:  # nothing cached
            return {}
        
        return self._train_vid_buf.retrieve_frames(
            env_idxs=env_idxs,
        )

    def on_rollout_end(self):
        self._train_vid_buf.on_rollout_end()

    def launch_evaluation_during_training(
        self,
        ckpt_idx: int
    ):
        self._eval_monitor.launch_evaluation_during_training(
            ckpt_idx=ckpt_idx
        )

    def retrieve_eval_results_during_training(self):
        if self._disable_logging:  # no eval results written
            return {}

        return self._eval_monitor.retrieve_eval_results_during_training()
    
    def wait_for_eval_results_after_training(
        self,
        writer: WeightsAndBiasesWriter,
        sleep_sec: float = 10.0,
    ):
        if self._disable_logging:  # no eval results written, don't wait
            return

        while self._eval_monitor.has_pending_eval_results():
            aggregated_stats = self._eval_monitor.retrieve_eval_results_during_training()

            if len(aggregated_stats) > 0:
                writer.add_aggregated_logs(
                    aggregated_stats,
                    step_id=None  # just commit
                )

            time.sleep(sleep_sec)

    def launch_evaluation_after_training(
        self,
        writer: WeightsAndBiasesWriter,
    ):
        self._eval_monitor.launch_evaluation_after_training(
            writer=writer,
        )
