import random
import importlib
from pydantic import BaseModel

from modules.configuration_models import (
    SpaceConfig,
    AgentConfig,
    OperatingArea,
    DynamicTaskGenerationConfig,
)
from modules.task import Task
from modules.agent import Agent


def generate_positions(quantity, x_min, x_max, y_min, y_max, radius=10):
    positions = []
    while len(positions) < quantity:
        pos = (
            random.uniform((x_min + radius), (x_max - radius)),
            random.uniform((y_min + radius), (y_max - radius)),
        )
        if radius > 0:
            if all(
                (abs(pos[0] - p[0]) > radius and abs(pos[1] - p[1]) > radius)
                for p in positions
            ):
                positions.append(pos)
        else:
            positions.append(pos)
    return positions


def generate_tasks(
    config: SpaceConfig, num_tasks: int, task_id_start: int
) -> list[Task]:

    tasks_positions = generate_positions(
        num_tasks,
        config.tasks.locations.x_min,
        config.tasks.locations.x_max,
        config.tasks.locations.y_min,
        config.tasks.locations.y_max,
        radius=config.tasks.locations.non_overlap_radius,
    )

    tasks = []
    for idx, pos in enumerate(tasks_positions):
        amount = random.uniform(config.tasks.amounts.min, config.tasks.amounts.max)
        radius = max(1, amount / config.simulation.task_visualisation_factor)
        tasks.append(Task(idx + task_id_start, pos, radius, amount))

    return tasks


def generate_agents(
    tasks: list[Task], config: SpaceConfig, strategy: str
) -> list[Agent]:

    positions = generate_positions(
        config.agents.quantity,
        config.agents.locations.x_min,
        config.agents.locations.x_max,
        config.agents.locations.y_min,
        config.agents.locations.y_max,
        radius=config.agents.locations.non_overlap_radius,
    )
    bounds: OperatingArea = config.tasks.locations
    agent_config: AgentConfig = config.agents

    # Import selected task assignment type and its config model
    assignment_class, assignment_config_class = load_task_assignment_types(
        config.decision_making, strategy
    )

    # Configuration object for task assignment class
    assignment_config: BaseModel = assignment_config_class(
        **config.decision_making[strategy]
    )

    agents = []
    for agent_id, position in enumerate(positions):
        tasker = assignment_class(agent_id, assignment_config, agent_config)
        agent = Agent(agent_id, position, tasks, tasker, bounds, agent_config)
        agents.append(agent)

    # TODO investigate possibility of removing all_agents
    for agent in agents:
        agent.all_agents = agents

    return agents


def load_task_assignment_types(config_dict: dict, strategy: str) -> tuple[type, type]:
    """Returns the task assignment class and its accompanying configuration class
    from the plugin module selected by `strategy`.
    """
    if strategy not in config_dict:
        raise ValueError(
            f"Unrecognized strategy {strategy}. Options: {list(config_dict.keys())}"
        )
    module_path, class_name = config_dict[strategy]["plugin"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    decision_class: type = getattr(module, class_name)
    decision_config_class: type = getattr(module, class_name + "Config")
    return (decision_class, decision_config_class)


class DynamicTaskGenerator:
    """Periodically adds new tasks to the simulation."""

    def __init__(self, config: DynamicTaskGenerationConfig):
        self.generation_count: int = 0
        self.last_generation_time: float = 0
        self.config = config
        self.enabled = config.enabled

    def add_new_tasks(self, num_tasks: int, tasks: list[Task]):
        new_task_id_start = len(tasks)
        new_tasks = generate_tasks(self.config, num_tasks, new_task_id_start)
        tasks.extend(new_tasks)

    def update(self, sim_time: float, tasks: list[Task]):
        if self.done():
            return

        if sim_time - self.last_generation_time >= self.config.interval_seconds:
            self.add_new_tasks(self.config.tasks_per_generation, tasks)
            self.last_generation_time = sim_time
            self.generation_count += 1

            print(
                f"[{sim_time:.2f}] "
                f"Added {self.config.tasks_per_generation} new tasks: "
                f"Generation {self.generation_count}."
            )

    def done(self) -> bool:
        if not self.enabled:
            return True
        if self.generation_count == self.config.max_generations:
            return True
        return False
