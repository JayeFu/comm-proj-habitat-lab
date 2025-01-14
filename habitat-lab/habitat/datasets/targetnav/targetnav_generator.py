# Adapted from
# https://github.com/facebookresearch/habitat-lab/blob/v0.2.3/habitat-lab/habitat/datasets/pointnav/pointnav_generator.py

from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from habitat.core.simulator import ShortestPathPoint
from habitat.datasets.pointnav.pointnav_generator import (
    ISLAND_RADIUS_LIMIT, is_compatible_episode)
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import NavigationGoal, TargetNavigationEpisode

try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException
try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    habitat_sim = BaseException


def _create_targetnav_episode(
    episode_id: Union[int, str],
    scene_id: str,
    target_id: str,
    target_scale: List[float], 
    scene_layout_id: str,
    start_position: List[float],
    start_rotation: List[float],
    target_position: List[float],
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None,
    radius: Optional[float] = None,
    info: Optional[Dict[str, float]] = None,
) -> Optional[TargetNavigationEpisode]:
    goals = [NavigationGoal(position=target_position, radius=radius)]
    return TargetNavigationEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        target_id=target_id,
        target_scale=target_scale,
        scene_layout_id=scene_layout_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )


def generate_targetnav_episodes(
    sim: "HabitatSim",
    target_id: str,
    target_scale_range: Tuple[float, float],
    scene_layout_id: str,
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
) -> Generator[TargetNavigationEpisode, None, None]:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param scene_layout_id: 
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        target_position = sim.sample_navigable_point()

        while target_position[1] > 0.5:
            target_position = sim.sample_navigable_point()

        target_scale = np.random.uniform(*target_scale_range)

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        for _retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            if is_compatible:
                break
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            shortest_paths = None
            if is_gen_shortest_path:
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue

            episode = _create_targetnav_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.scene,
                target_id=target_id,
                target_scale=[target_scale] * 3,
                scene_layout_id=scene_layout_id,
                start_position=source_position,
                start_rotation=source_rotation,
                target_position=target_position,
                shortest_paths=shortest_paths,
                radius=shortest_path_success_distance,
                info={"geodesic_distance": dist},
            )

            episode_count += 1
            yield episode

