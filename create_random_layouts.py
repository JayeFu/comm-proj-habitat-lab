import argparse
import json
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

import habitat
from habitat.utils.runtime_objs import RunTimeObjectManager

DEFAULT_SCALE = [1.0, 1.0, 1.0]
ALL_ASSETS = [
    {
        "asset": "data/objects/obstacles/carton_box.glb",
        "scale": DEFAULT_SCALE,
        "height": 0.2,
        "max_num": 5,
    },
    {
        "asset": "data/objects/obstacles/carton_box2.glb",
        "scale": DEFAULT_SCALE,
        "height": 0.16,
        "max_num": 5,
    },
    {
        "asset": "data/objects/obstacles/office_chair.glb",
        "scale": DEFAULT_SCALE,
        "height": 0.4,
        "max_num": 3,
    },
    {
        "asset": "data/objects/obstacles/shelf.glb",
        "scale": DEFAULT_SCALE,
        "height": 0.9,
        "max_num": 3,
    },
]


def create_layout_config(
    scene_template_fp: str,
    num_obstacles: 10,
    save_dir: str,
    layout_idx: int,
):
    print(f"Create layout {layout_idx} for {scene_template_fp}")
    cfg = habitat.get_config(
        "benchmark/nav/targetnav/targetnav_test.yaml"
    )
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = scene_template_fp

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)
    runtime_obj_mgr = RunTimeObjectManager(sim=sim)
    obs_cntr = [0] * len(ALL_ASSETS)

    scene_layout_cfg = {"walls": [], "obstacles": []}

    for _ in range(num_obstacles):
        ind = np.random.randint(len(ALL_ASSETS))

        while obs_cntr[ind] > ALL_ASSETS[ind]["max_num"]:
            ind = np.random.randint(len(ALL_ASSETS))

        obs_cntr[ind] += 1

        asset_cfg = ALL_ASSETS[ind]

        obs_pos = sim.sample_navigable_point()
        while obs_pos[1] > 0.5:
            obs_pos = sim.sample_navigable_point()

        rot_amount = np.random.uniform(0, 2 * np.pi)
        rot_vector = [0.0, 1.0, 0.0]

        single_layout_cfg = [{
            "asset": asset_cfg["asset"],
            "pos": [obs_pos[0], asset_cfg["height"], obs_pos[2]],
            "rot_amount": rot_amount,
            "rot_vector": rot_vector,
            "scale": asset_cfg["scale"],
        }]
        runtime_obj_mgr.add_obstacles(
            obstacles_layout_cfg=single_layout_cfg
        )

        runtime_obj_mgr.recompute_navmesh()

        scene_layout_cfg["obstacles"].extend(single_layout_cfg)

    out_file = os.path.join(save_dir, f"layout_{layout_idx}.json")
    with open(out_file, "w") as f:
        json.dump(scene_layout_cfg, f)
    print(f"Save layout {layout_idx} to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-template-fp",
        type=str,
        required=True,
        help="Scene template filepath",
    )
    parser.add_argument(
        "--num-layouts",
        type=int,
        default=50,
        help="Number of layouts to generate",
    )
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=10,
        help="Number of obstacles to generate",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--layout-name",
        type=str,
        required=True,
        help="Layout name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name",
    )
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.layout_name, args.split)
    os.makedirs(save_dir, exist_ok=True)

    gen_args_l = [
        (args.scene_template_fp, args.num_obstacles, save_dir, layout_idx)
        for layout_idx in range(args.num_layouts)
    ]
    with multiprocessing.Pool(8) as pool, tqdm(total=len(gen_args_l)) as pbar:
        for _ in pool.starmap(create_layout_config, gen_args_l):
            pbar.update()


if __name__ == "__main__":
    main()
