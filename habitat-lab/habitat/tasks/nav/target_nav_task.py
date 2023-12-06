from typing import Any, Optional, TYPE_CHECKING

from habitat.core.embodied_task import (
    EmbodiedTask,
)
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.datasets.targetnav.targetnav_dataset import TargetNavDatasetV0
from habitat.tasks.nav.nav import (
    DistanceToGoal,
    EuclideanDistanceToGoal,
    EuclideanSuccess,
    SPL,
    NavigationTask,
    TargetNavigationEpisode
)
from habitat.utils.runtime_objs import RunTimeObjectManager

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class NonStopEuclideanSuccess(EuclideanSuccess):
    r"""Whether or not the agent succeeded at its task without requiring stop action

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "nonstop_euclidean_success"

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            EuclideanDistanceToGoal.cls_uuid
        ].get_metric()

        if distance_to_target < self._config.success_distance:
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class NonStopEuclideanSPL(SPL):
    r"""Non-stop Euclidean SPL

    Similar to spl with Euclidean success and no stop action
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "nonstop_euclidean_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DistanceToGoal.cls_uuid,
             EuclideanDistanceToGoal.cls_uuid,
             NonStopEuclideanSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_nonstop_euclidean_success = \
            task.measurements.measures[NonStopEuclideanSuccess.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_nonstop_euclidean_success * (
            self._start_end_episode_distance
            / max(
            self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_task(name="TargetNav-v0")
class TargetNavigationTask(NavigationTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional["TargetNavDatasetV0"] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self._runtime_obj_mgr = RunTimeObjectManager(sim=sim)
        self._prev_scene_layout_id: Optional[str] = None

    def _add_target_to_goal_pos(self, episode: TargetNavigationEpisode):
        self._runtime_obj_mgr.delete_added_target()
        target_position = episode.goals[0].position.copy()
        target_position[1] = 1.5  # hardcode to agent height
        self._runtime_obj_mgr.add_target(
            target_path=episode.target_id,
            position=target_position,
        )

    def _add_layout_to_scene(self, episode: TargetNavigationEpisode):
        if self._prev_scene_layout_id is None \
           or episode.scene_layout_id != self._prev_scene_layout_id:
            # Clear previous scene layout
            self._runtime_obj_mgr.delete_added_walls()
            self._runtime_obj_mgr.delete_added_obstacles()

            # Load new scene layout
            self._runtime_obj_mgr.add_walls(
                walls_layout_cfg=episode.scene_layout['walls']
            )
            self._runtime_obj_mgr.add_obstacles(
                obstacles_layout_cfg=episode.scene_layout['obstacles']
            )

            # Cache new scene layout id
            self._prev_scene_layout_id = episode.scene_layout_id

    def reset(self, episode: TargetNavigationEpisode):
        self._runtime_obj_mgr.refresh(sim=self._sim)
        self._add_target_to_goal_pos(episode=episode)
        self._add_layout_to_scene(episode=episode)
        # Recompute navmesh after adding
        self._runtime_obj_mgr.recompute_navmesh()

        return super().reset(episode=episode)

