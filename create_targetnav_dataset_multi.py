import argparse
import glob
import gzip
import json
import multiprocessing
import numpy as np
import os
from typing import Tuple

from tqdm import tqdm

import habitat
from habitat.datasets.targetnav.targetnav_generator import (
    generate_targetnav_episodes,
)
from habitat.utils.runtime_objs import RunTimeObjectManager


def _generate_fn(
    scene_template_fp: str, layout_dir: str, target_dir: str, 
    target_scale_range: Tuple[float, float],
    target_height_range: Tuple[float, float],
    out_dir: str, env_idx: int,
    num_episodes: int
):
    print(f"Create targetnav dataset for {scene_template_fp}")
    cfg = habitat.get_config(
        "benchmark/nav/targetnav/targetnav_test.yaml"
    )
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = scene_template_fp

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)
    runtime_obj_mgr = RunTimeObjectManager(sim=sim)

    all_layout_fp_l = glob.glob(os.path.join(layout_dir, '*.json'))
    all_layout_fp_l.sort()

    all_target_fp_l = glob.glob(os.path.join(target_dir, '*.glb'))
    all_target_fp_l = [fp for fp in all_target_fp_l if 'collision' not in fp]

    dset = habitat.datasets.make_dataset("TargetNav-v0")
    for _ in range(num_episodes):
        layout_ind = np.random.randint(len(all_layout_fp_l))
        layout_fp = all_layout_fp_l[layout_ind]
        with open(layout_fp, 'r') as f:
            layout_cfg = json.load(f)
        
        target_ind = np.random.randint(len(all_target_fp_l))
        target_fp = all_target_fp_l[target_ind]

        # Delete previous walls and obstacles
        runtime_obj_mgr.delete_added_obstacles()
        runtime_obj_mgr.delete_added_walls()
        # Add walls
        runtime_obj_mgr.add_walls(walls_layout_cfg=layout_cfg['walls'])
        # Add obstacles
        runtime_obj_mgr.add_obstacles(obstacles_layout_cfg=layout_cfg['obstacles'])
        runtime_obj_mgr.recompute_navmesh()

        dset.episodes.extend(
            generate_targetnav_episodes(
                sim=sim,
                target_id=target_fp,
                target_scale_range=target_scale_range,
                scene_layout_id=layout_fp,
                num_episodes=1,
                is_gen_shortest_path=False,
            )
        )

    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/"):]
        ep.goals[0].position[1] = np.random.uniform(*target_height_range)

    out_file = os.path.join(out_dir, f"env_{env_idx}.json.gz")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    print(f"Save episodes for {layout_fp} to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene_dir', type=str, required=True,
        help='Directory containing scene templates'
    )
    parser.add_argument(
        '--layout_dir', type=str, required=True,
        help='Directory containing scene layouts'
    )
    parser.add_argument(
        '--dset_name', type=str, required=True,
        help='Name of dataset'
    )
    parser.add_argument(
        '--dset_version', type=str, required=True,
        help='Version of dataset'
    )
    parser.add_argument(
        '--target_dir', type=str, required=True,
        help='Directory containing target objects'
    )
    parser.add_argument(
        '--target_scale_range', type=float, nargs=2, required=True,
        help='Range of target scale'
    )
    parser.add_argument(
        '--target_height_range', type=float, nargs=2, required=True,
        help='Range of target height'
    )
    parser.add_argument(
        "--num_envs", type=int, default=10,
        help='Number of environments to create'
    )
    parser.add_argument(
        "--num_episodes", type=int, default=int(1e4),
        help='Number of episodes to create for each environment'
    )
    parser.add_argument(
        "--split", type=str, default='train'
    )
    args = parser.parse_args()

    scene_template_fp_l = glob.glob(os.path.join(args.scene_dir, '*.glb'))
    assert args.num_envs % len(scene_template_fp_l) == 0, \
        f"Number of environments ({args.num_envs}) must be divisible by number of scene templates ({len(scene_template_fp_l)})"

    scene_template_fp_l = scene_template_fp_l * (args.num_envs // len(scene_template_fp_l))

    out_dir = f"./data/datasets/targetnav/{args.dset_name}/{args.dset_version}/{args.split}/"
    os.makedirs(out_dir, exist_ok=True)
    scene_out_dir = os.path.join(out_dir, 'content')
    os.makedirs(scene_out_dir, exist_ok=True)

    gen_args_l = [
        (scene_template_fp, args.layout_dir, args.target_dir, args.target_scale_range, args.target_height_range, scene_out_dir, idx, args.num_episodes)
        for idx, scene_template_fp in enumerate(scene_template_fp_l)
    ]

    with multiprocessing.Pool(8) as pool, tqdm(total=len(scene_template_fp_l)) as pbar:
        for _ in pool.starmap(_generate_fn, gen_args_l):
            pbar.update()

    empty_dset_fp = os.path.join(out_dir, f'{args.split}.json.gz')
    with gzip.open(empty_dset_fp, 'wt') as f:
        json.dump(dict(episodes=[]), f)


if __name__ == "__main__":
    main()
