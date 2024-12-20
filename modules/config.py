from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    PositiveFloat,
    NonNegativeFloat,
)
from typing import Any
from pathlib import Path
from enum import Enum

"""Pydantic models for SPACE simulation configuration"""


class RenderingMode(str, Enum):
    Screen = "Screen"
    Terminal = "Terminal"
    Headless = "None"


class FloatRange(BaseModel):
    min: float
    max: float


class OperatingArea(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    non_overlap_radius: NonNegativeFloat


class DynamicTaskGenerationConfig(BaseModel):
    enabled: bool = Field(default=False)
    interval_seconds: PositiveInt = Field(default=10)
    max_generations: PositiveInt = Field(default=5)
    tasks_per_generation: PositiveInt = Field(default=5)


class RenderingOptions(BaseModel):
    agent_tail: bool = Field(default=False)
    agent_communication_topology: bool = Field(default=False)
    agent_situation_awareness_circle: bool = Field(default=False)
    agent_id: bool = Field(default=False)
    agent_work_done: bool = Field(default=False)
    agent_assigned_task_id: bool = Field(default=False)
    agent_path_to_assigned_tasks: bool = Field(default=False)
    task_id: bool = Field(default=False)


class SavingOptions(BaseModel):
    output_folder: Path
    with_date_subfolder: bool
    save_gif: bool = Field(default=False)  # Only works if `rendering_mode` is `Screen`
    save_timewise_result_csv: bool = Field(default=False)
    save_agentwise_result_csv: bool = Field(default=False)
    save_config_yaml: bool = Field(default=False)


class AgentConfig(BaseModel):
    behavior_tree_xml: Path
    quantity: PositiveInt
    locations: OperatingArea
    max_speed: PositiveFloat
    max_accel: PositiveFloat
    max_angular_speed: PositiveFloat
    target_approaching_radius: PositiveFloat
    # work rate for each agent (per 1.0/simulation.sampling_freq)
    work_rate: PositiveFloat
    communication_radius: PositiveFloat
    situation_awareness_radius: PositiveFloat  # 0 represents "global", meaning that each agent can access to the information of all the tasks
    random_exploration_duration: PositiveFloat  # sec


class TaskConfig(BaseModel):
    quantity: PositiveInt
    locations: OperatingArea
    threshold_done_by_arrival: PositiveFloat
    amounts: FloatRange
    dynamic_task_generation: DynamicTaskGenerationConfig


class SimConfig(BaseModel):
    sampling_freq: PositiveFloat

    # 0 mean max booster; 1 means normal; 10 means 10-times faster
    speed_up_factor: NonNegativeInt

    # 0 means no limit
    max_simulation_time: NonNegativeFloat
    agent_track_size: NonNegativeInt
    screen_width: PositiveInt
    screen_height: PositiveInt
    gif_recording_fps: PositiveFloat

    # visualization factor for tasks : 10 means converting 10 amount to 1 pixel
    task_visualisation_factor: PositiveInt
    profiling_mode: bool
    rendering_mode: RenderingMode

    # Only works if `rendering_mode` is `Screen`
    rendering_options: RenderingOptions
    saving_options: SavingOptions


class SpaceConfig(BaseModel):
    agents: AgentConfig
    tasks: TaskConfig
    simulation: SimConfig
    decision_making: Any  # Custom config


if __name__ == "__main__":
    import yaml

    for yaml_file in Path("../config/example").glob("*.yaml"):
        print(yaml_file)
        with open(yaml_file) as f:
            config_dict = yaml.safe_load(f)
            myconf = SpaceConfig(**config_dict)
