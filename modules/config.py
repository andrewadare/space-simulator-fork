from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    PositiveFloat,
    NonNegativeFloat,
)
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


class DynamicTaskGeneration(BaseModel):
    enabled: bool
    interval_seconds: PositiveInt
    max_generations: PositiveInt
    tasks_per_generation: PositiveInt


class RenderingOptions(BaseModel):
    agent_tail: bool
    agent_communication_topology: bool
    agent_situation_awareness_circle: bool
    agent_id: bool
    agent_work_done: bool
    agent_assigned_task_id: bool
    agent_path_to_assigned_tasks: bool
    task_id: bool


class SavingOptions(BaseModel):
    output_folder: Path
    with_date_subfolder: bool
    save_gif: bool  # Only works if `rendering_mode` is `Screen`
    save_timewise_result_csv: bool
    save_agentwise_result_csv: bool
    save_config_yaml: bool


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
    dynamic_task_generation: DynamicTaskGeneration


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
    decision_making: dict


if __name__ == "__main__":
    import yaml

    for yaml_file in Path("../config/example").glob("*.yaml"):
        print(yaml_file)
        with open(yaml_file) as f:
            config_dict = yaml.safe_load(f)
            myconf = SpaceConfig(**config_dict)
