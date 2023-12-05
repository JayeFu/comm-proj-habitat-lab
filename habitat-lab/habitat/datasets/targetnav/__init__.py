#!/usr/bin/env python3

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


# This is a result of moving SimulatorActions away from core
# and into simulators specifically. As a result of that the connection points
# for our tasks and datasets for actions is coming from inside habitat-sim
# which makes it impossible for anyone to use habitat-lab without having
# habitat-sim installed. In a future PR we will implement a base simulator
# action class which will be the connection point for tasks and datasets.
# Post that PR we would no longer need try register blocks.
def _try_register_targetnavdatasetv0():
    try:
        from habitat.datasets.targetnav.targetnav_dataset import (  # noqa: F401
            TargetNavDatasetV0,
        )

    except ImportError as e:
        targetnav_import_error = e

        @registry.register_dataset(name="TargetNav-v0")
        class TargetnavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise targetnav_import_error
