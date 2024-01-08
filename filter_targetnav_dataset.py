import argparse
import glob
import gzip
import json
import multiprocessing
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import habitat_sim
from habitat.utils.runtime_objs import RunTimeObjectManager


class DatasetFilter:
    def __init__(self):
        self._cfg_settings = \
            habitat_sim.utils.settings.default_sim_settings.copy()
        self._hab_cfg = habitat_sim.utils.settings.make_cfg(self._cfg_settings)
        self._hab_cfg.agents[0].height = 0.2
        self._hab_cfg.agents[0].radius = 0.17

        self._sim = habitat_sim.Simulator(self._hab_cfg)
        self._runtime_obj_mgr = RunTimeObjectManager(sim=self._sim)
        self._prev_scene_layout_id: Optional[str] = None

    def filter_dataset(self, dset_path: str, overwrite_now: bool = True):
        with gzip.open(dset_path, 'rt') as f:
            dset = json.load(f)

        episodes = dset['episodes']

        valid_eps, invalid_eps = [], []
        for ep in episodes:
            valid = self.check_episode(episode=ep)
            if valid:
                valid_eps.append(ep)
            else:
                invalid_eps.append(ep)

        dset['episodes'] = valid_eps
        if overwrite_now:
            with gzip.open(dset_path, 'wt') as f:
                json.dump(dset, f)

        num_total_eps, num_valid_eps = len(episodes), len(valid_eps)
        print(f"In total {num_total_eps} episodes")
        print(f"\t{num_valid_eps} valid episodes "
              f"({100 * num_valid_eps / num_total_eps:.2f} %)")

        return dset

    def check_episode(self, episode: Dict):
        self._cfg_settings['scene'] = self._correct_filepath(
            filepath=episode['scene_id']
        )
        self._hab_cfg = habitat_sim.utils.settings.make_cfg(self._cfg_settings)
        self._sim.reconfigure(self._hab_cfg)
        self._runtime_obj_mgr.refresh(sim=self._sim)

        self._add_target_to_goal(episode=episode)
        self._add_layout_to_scene(episode=episode)
        self._runtime_obj_mgr.recompute_navmesh()

        path = habitat_sim.ShortestPath()
        path.requested_start = episode['start_position']
        path.requested_end = episode['goals'][0]['position']
        self._sim.pathfinder.find_path(path)

        if not np.isfinite(path.geodesic_distance):
            return False
        else:
            return True

    def _add_target_to_goal(self, episode: Dict):
        self._runtime_obj_mgr.delete_added_target()

        target_pos = episode['goals'][0]['position'].copy()
        # target_pos[1] = 1.5
        self._runtime_obj_mgr.add_target(
            target_path=episode['target_id'],
            position=target_pos,
            scale=episode['target_scale'],
        )

    def _add_layout_to_scene(self, episode: Dict):
        if len(episode['scene_layout_id']) > 0 \
           and (self._prev_scene_layout_id is None \
           or episode['scene_layout_id'] != self._prev_scene_layout_id):
            # Clear previous scene layout
            self._runtime_obj_mgr.delete_added_walls()
            self._runtime_obj_mgr.delete_added_obstacles()

            # Load new scene layout
            scene_layout_fp = self._correct_filepath(
                filepath=episode['scene_layout_id']
            )
            with open(scene_layout_fp, 'r') as f:
                scene_layout_cfg = json.load(f)

            self._runtime_obj_mgr.add_walls(
                walls_layout_cfg=scene_layout_cfg['walls']
            )
            self._runtime_obj_mgr.add_obstacles(
                obstacles_layout_cfg=scene_layout_cfg['obstacles']
            )

            # Cache new scene layout id
            self._prev_scene_layout_id = episode['scene_layout_id']

    @staticmethod
    def _correct_filepath(filepath: str):
        if not osp.exists(filepath):  # Try adding prefix
            filepath = "./data/scene_datasets/" + filepath
            if not osp.exists(filepath):
                raise FileNotFoundError(f"Cannot find {filepath} ...")
        return filepath


def _filter_fn(dset_idx: int, dset_path: str, overwrite_now: bool = True):
    dset = DatasetFilter().filter_dataset(dset_path=dset_path, overwrite_now=overwrite_now)

    return dset_idx, dset


def reindex_datasets(datasets: List[Dict], dataset_files: List[str]):
    ep_count = 0
    for dset, dset_file in zip(datasets, dataset_files):
        for ep in dset['episodes']:
            ep['episode_id'] = ep_count  # reindex

            ep_count += 1

        with gzip.open(dset_file, 'wt') as f:
            json.dump(dset, f)


def filter_targetnav_datasets(
    scene_name: str, layout_name: str, split: str
):
    dataset_root_dir = f"./data/datasets/targetnav/{scene_name}/{layout_name}/{split}/"
    content_scenes_dir = osp.join(dataset_root_dir, 'content')

    dset_files = glob.glob(osp.join(content_scenes_dir, "*.json.gz"))
    dset_files.sort()  # Sort for indexing

    need_reindex = split != 'train'
    filter_args = [
        (idx, dset_path, not need_reindex)
        for idx, dset_path in enumerate(dset_files)
    ]
    results = []
    with multiprocessing.Pool(8) as pool, tqdm(total=len(dset_files)) as pbar:
        for res in pool.starmap(_filter_fn, filter_args):
            results.append(res)
            pbar.update()

    if need_reindex:  # not train
        def take_idx(l):
            return l[0]

        results.sort(key=take_idx)  # sort by env_idx
        reindex_datasets(
            datasets=[res[1] for res in results],
            dataset_files=dset_files
        )

def filter_multiple_scene_targetnav_datasets(
    dset_name: str, split: str
):
    dataset_root_dir = f"./data/datasets/targetnav/{dset_name}/{split}/"
    content_scenes_dir = osp.join(dataset_root_dir, 'content')
    dset_files = glob.glob(osp.join(content_scenes_dir, "*.json.gz"))
    dset_files.sort()  # Sort for indexing

    need_reindex = split != 'train'
    filter_args = [
        (idx, dset_path, not need_reindex)
        for idx, dset_path in enumerate(dset_files)
    ]
    results = []
    with multiprocessing.Pool(8) as pool, tqdm(total=len(dset_files)) as pbar:
        for res in pool.starmap(_filter_fn, filter_args):
            results.append(res)
            pbar.update()

    if need_reindex:  # not train
        def take_idx(l):
            return l[0]

        results.sort(key=take_idx)  # sort by env_idx
        reindex_datasets(
            datasets=[res[1] for res in results],
            dataset_files=dset_files
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiple_scenes", action="store_true",
        help="whether to filter multiple scene datasets"
    )

    parser.add_argument(
        "--scene_name", type=str, default=None,
        help="scene name"
    )
    parser.add_argument(
        "--layout_name", type=str, default=None,
        help="layout name"
    )

    parser.add_argument(
        "--dset_name", type=str, default=None,
        help="dataset name"
    )

    parser.add_argument(
        "--split", type=str, required=True,
        help="dataset split"
    )
    args = parser.parse_args()

    if args.multiple_scenes:
        assert args.dset_name is not None

        filter_multiple_scene_targetnav_datasets(
            dset_name=args.dset_name,
            split=args.split
        )
    else:
        assert args.scene_name is not None and args.layout_name is not None

        filter_targetnav_datasets(
            scene_name=args.scene_name,
            layout_name=args.layout_name,
            split=args.split
        )


if __name__ == '__main__':
    main()
