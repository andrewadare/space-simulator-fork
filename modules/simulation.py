import random


from modules.configuration_models import SpaceConfig
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
        radius = amount / config.simulation.task_visualisation_factor
        tasks.append(Task(idx + task_id_start, pos, radius, amount))

    return tasks


def generate_agents(tasks: list[Task], config: SpaceConfig):

    agents_positions = generate_positions(
        config.agents.quantity,
        config.agents.locations.x_min,
        config.agents.locations.x_max,
        config.agents.locations.y_min,
        config.agents.locations.y_max,
        radius=config.agents.locations.non_overlap_radius,
    )

    agents = [
        Agent(idx, pos, tasks, config) for idx, pos in enumerate(agents_positions)
    ]

    for agent in agents:
        agent.all_agents = agents  # TODO see README

    return agents
