#!/usr/bin/env python3

import json
import os
from typing import List, Optional

from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import (
    PointNavDatasetV1,
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX
)
from habitat.tasks.nav.nav import (
    NavigationGoal,
    ShortestPathPoint,
    TargetNavigationEpisode,
)


@registry.register_dataset(name="TargetNav-v0")
class TargetNavDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDatasetV1 that loads Target Navigation dataset."""

    episodes: List[TargetNavigationEpisode]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = TargetNavigationEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            if len(episode.scene_layout_id) > 0:
                if not os.path.exists(episode.scene_layout_id):
                    raise FileNotFoundError(f"Cannot find {episode.scene_layout_id}")
                
                with open(episode.scene_layout_id) as f:
                    episode.scene_layout = json.load(f)

            self.episodes.append(episode)