# Adapted from
# https://github.com/facebookresearch/habitat-lab/blob/v0.2.3/habitat-baselines/habitat_baselines/rl/ddppo/data_generation/create_gibson_large_dataset.py

import argparse
import gzip
import json
import multiprocessing
import os
from os import path as osp

from tqdm import tqdm

import habitat
from habitat.datasets.targetnav.targetnav_generator import (
    generate_targetnav_episodes,
)
from habitat.utils.runtime_objs import RunTimeObjectManager


def _generate_fn(
        scene_template_fp: str, layout_fp: str, target_fp: str,
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
    with open(layout_fp, 'r') as f:
        layout_cfg = json.load(f)

    # Add walls
    runtime_obj_mgr.add_walls(walls_layout_cfg=layout_cfg['walls'])
    # Add obstacles
    runtime_obj_mgr.add_obstacles(obstacles_layout_cfg=layout_cfg['obstacles'])
    runtime_obj_mgr.recompute_navmesh()

    dset = habitat.datasets.make_dataset("TargetNav-v0")
    dset.episodes = list(
        generate_targetnav_episodes(
            sim=sim,
            target_id=target_fp,
            scene_layout_id=layout_fp,
            num_episodes=num_episodes,
            is_gen_shortest_path=False,
        )
    )

    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/"):]

    out_file = osp.join(out_dir, f"env_{env_idx}.json.gz")

    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    print(f"Save episodes for {layout_fp} to {out_file}")


def generate_targetnav_dataset(
        scene_template_fp: str, layout_dir: str, target_fp: str,
        num_episodes: int = int(1e4), split: str = 'train'
):
    scene_name = scene_template_fp.split('/')[-1].split('.')[0]
    layout_name = layout_dir.split('/')[-1]
    layout_dir = osp.join(layout_dir, split)

    layout_fp_l = []
    for layout_fn in sorted(os.listdir(layout_dir)):
        if layout_fn.endswith('.json'):
            layout_fp = osp.join(layout_dir, layout_fn)
            layout_fp_l.append(layout_fp)
    print(f"Total number of layouts: {len(layout_fp_l)}")

    out_dir = f"./data/datasets/targetnav/{scene_name}/{layout_name}/{split}/"
    os.makedirs(out_dir, exist_ok=True)
    scene_out_dir = osp.join(out_dir, 'content')
    os.makedirs(scene_out_dir, exist_ok=True)

    gen_args_l = [
        (scene_template_fp, layout_fp, target_fp, scene_out_dir, idx, num_episodes)
        for idx, layout_fp in enumerate(layout_fp_l)
    ]
    with multiprocessing.Pool(8) as pool, tqdm(total=len(layout_fp_l)) as pbar:
        for _ in pool.starmap(_generate_fn, gen_args_l):
            pbar.update()

    empty_dset_fp = osp.join(out_dir, f'{split}.json.gz')
    with gzip.open(empty_dset_fp, 'wt') as f:
        json.dump(dict(episodes=[]), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_template", type=str, required=True)
    parser.add_argument("--layouts", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=int(1e4))
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args()

    generate_targetnav_dataset(
        scene_template_fp=args.scene_template,
        layout_dir=args.layouts,
        target_fp=args.target,
        num_episodes=args.num_episodes,
        split=args.split
    )


if __name__ == '__main__':
    main()

