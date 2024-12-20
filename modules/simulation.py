import random
from math import floor, ceil

from modules.configuration_models import SpaceConfig, AgentConfig
from modules.task import Task
from modules.agent import Agent


def generate_positions(quantity, x_min, x_max, y_min, y_max, radius=10):
    positions = []
    while len(positions) < quantity:
        pos = (
            random.randint(ceil(x_min + radius), floor(x_max - radius)),
            random.randint(ceil(y_min + radius), floor(y_max - radius)),
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


# TODO fix up task_quantity fallback logic
def generate_tasks(
    config: SpaceConfig, task_quantity: int | None = None, task_id_start: int = 0
) -> list[Task]:
    if task_quantity is None:
        task_quantity = config.tasks.quantity

    tasks_positions = generate_positions(
        config.tasks.quantity,
        config.tasks.locations.x_min,
        config.tasks.locations.x_max,
        config.tasks.locations.y_min,
        config.tasks.locations.y_max,
        radius=config.tasks.locations.non_overlap_radius,
    )

    tasks = []
    for idx, pos in enumerate(tasks_positions):
        amount = random.uniform(config.tasks.amounts.min, config.tasks.amounts.max)
        radius = amount / config.simulation.task_visualisation_factor
        tasks.append(Task(idx + task_id_start, pos, radius, amount))

    return tasks


def generate_agents(tasks: list[Task], config: AgentConfig):

    agents_positions = generate_positions(
        config.quantity,
        config.locations.x_min,
        config.locations.x_max,
        config.locations.y_min,
        config.locations.y_max,
        radius=config.locations.non_overlap_radius,
    )

    agents = [
        Agent(idx, pos, tasks, config) for idx, pos in enumerate(agents_positions)
    ]

    for agent in agents:
        agent.all_agents = agents  # TODO see README
        agent.create_behavior_tree()  # TODO why not call this in constructor?

    return agents
